import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
from io import BytesIO
from typing import Dict, List, Optional
import altair as alt
import time


# NEW: Auth / DB
from authlib.integrations.requests_client import OAuth2Session
from supabase import create_client
from jose import jwt
import secrets


def _safe_rerun():
    """Streamlit compatibility: use st.rerun() if available; otherwise st.experimental_rerun()."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# =============================
# Page config
# =============================
st.set_page_config(page_title="DIET APP · Meal Planner", layout="wide")

# === Global styles (titles #BB4430) + tooltip CSS ===
st.markdown("""
<style>
  h1, h2, h3 { color: #BB4430 !important; }
  /* Avoid overlap on expander header */
  div.streamlit-expanderHeader { white-space: normal !important; overflow: hidden; }
  div.streamlit-expanderHeader p { margin: 0 !important; }

  /* Tooltip */
  .tooltip-wrap{position:relative;display:inline-block;cursor:help;}
  .tooltip-wrap .tooltip-content{
    visibility:hidden;opacity:0;transition:opacity .2s ease;
    position:absolute;left:0;top:120%;z-index:10000;
    background:rgba(0,0,0,0.9);color:#fff;border:1px solid rgba(255,255,255,.1);
    border-radius:8px;padding:10px 12px;width:360px;max-width:80vw;
    box-shadow:0 6px 24px rgba(0,0,0,.3);
  }
  .tooltip-wrap:hover .tooltip-content{visibility:visible;opacity:1;}
  .tooltip-content table{width:100%;border-collapse:collapse;font-size:0.85rem;}
  .tooltip-content th,.tooltip-content td{
    padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.12);text-align:left;
  }
  .tooltip-content th:nth-child(2),.tooltip-content td:nth-child(2){text-align:center;width:90px;}
</style>
""", unsafe_allow_html=True)

st.title("APP Recipe Builder")


# =============================
# Utilities
# =============================

def _norm_txt(s: str) -> str:
    """Normalize text for column matching: lowercase, remove accents, collapse spaces."""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float_series(x: pd.Series) -> pd.Series:
    """Convert a Series to float robustly (supports '3,4', spaces, etc.)."""
    if x.dtype.kind in {"i", "u", "f"}:
        return x.astype(float)
    return pd.to_numeric(
        x.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def kcal_from_macros(carb_g: float, prot_g: float, fat_g: float) -> float:
    return carb_g * 4.0 + prot_g * 4.0 + fat_g * 9.0


def mifflin_st_jeor_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    sex_n = _norm_txt(sex)
    male_tokens = {"hombre", "masculino", "varon", "varón", "male", "man"}
    if sex_n in male_tokens:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


# =============================
# Excel loading & normalization
# =============================

EXPECTED_NAME_KEYS = ["producto", "alimento", "nombre"]
EXPECTED_BRAND_KEYS = ["marca"]
EXPECTED_CAT_KEYS = ["categoria", "categoría"]
EXPECTED_SUBCAT_KEYS = ["subcategoria", "subcategoría"]

KCAL_PER_G_KEYS = [
    "energia (kcal/g)",
    "energia kcal/g",
    "calorias (kcal/g)",
    "calorias kcal/g",
    "kcal/g",
]
KCAL_PER_100G_KEYS = [
    "energia (kcal/100g)",
    "energia kcal/100g",
    "calorias (kcal/100g)",
    "calorias kcal/100g",
    "kcal/100g",
]

CARB_PER_G_KEYS = ["carbohidratos (g/g)", "hidratos (g/g)", "carbs (g/g)"]
CARB_PER_100G_KEYS = ["carbohidratos (g/100g)", "hidratos (g/100g)", "carbs (g/100g)"]

PROT_PER_G_KEYS = ["proteinas (g/g)", "proteínas (g/g)", "protein (g/g)"]
PROT_PER_100G_KEYS = ["proteinas (g/100g)", "proteínas (g/100g)", "protein (g/100g)"]

FAT_PER_G_KEYS = ["grasas (g/g)", "lipidos (g/g)"]
FAT_PER_100G_KEYS = ["grasas (g/100g)", "lipidos (g/100g)"]


def _find_first(cols_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in cols_map:
            return cols_map[key]
    return None


def _normalize_food_df(df: pd.DataFrame) -> pd.DataFrame:
    cols_map = {_norm_txt(c): c for c in df.columns}

    name_col = _find_first(cols_map, EXPECTED_NAME_KEYS)
    if not name_col:
        raise ValueError("Couldn't find a product name column (e.g., 'Producto'/'Alimento').")
    brand_col = _find_first(cols_map, EXPECTED_BRAND_KEYS)
    cat_col = _find_first(cols_map, EXPECTED_CAT_KEYS)
    subcat_col = _find_first(cols_map, EXPECTED_SUBCAT_KEYS)

    kcal_g = None
    kcal_g_src = _find_first(cols_map, KCAL_PER_G_KEYS)
    kcal_100_src = _find_first(cols_map, KCAL_PER_100G_KEYS)
    if kcal_g_src:
        kcal_g = _to_float_series(df[kcal_g_src])
    elif kcal_100_src:
        kcal_g = _to_float_series(df[kcal_100_src]) / 100.0

    carb_g = None
    crg = _find_first(cols_map, CARB_PER_G_KEYS)
    cr100 = _find_first(cols_map, CARB_PER_100G_KEYS)
    if crg:
        carb_g = _to_float_series(df[crg])
    elif cr100:
        carb_g = _to_float_series(df[cr100]) / 100.0

    prot_g = None
    prg = _find_first(cols_map, PROT_PER_G_KEYS)
    pr100 = _find_first(cols_map, PROT_PER_100G_KEYS)
    if prg:
        prot_g = _to_float_series(df[prg])
    elif pr100:
        prot_g = _to_float_series(df[pr100]) / 100.0

    fat_g = None
    frg = _find_first(cols_map, FAT_PER_G_KEYS)
    fr100 = _find_first(cols_map, FAT_PER_100G_KEYS)
    if frg:
        fat_g = _to_float_series(df[frg])
    elif fr100:
        fat_g = _to_float_series(df[fr100]) / 100.0

    clean = pd.DataFrame({
        "Producto": df[name_col].astype(str),
        "Marca": df[brand_col].astype(str) if brand_col else "",
        "Categoría": df[cat_col].astype(str) if cat_col else "",
        "Subcategoría": df[subcat_col].astype(str) if subcat_col else "",
        "carb_g": carb_g,
        "prot_g": prot_g,
        "fat_g": fat_g,
        "kcal_g": kcal_g,
    })

    if clean["kcal_g"].isna().any():
        has_macros = clean[["carb_g", "prot_g", "fat_g"]].notna().all(axis=1)
        clean.loc[has_macros & clean["kcal_g"].isna(), "kcal_g"] = clean.loc[
            has_macros & clean["kcal_g"].isna(), ["carb_g", "prot_g", "fat_g"]
        ].apply(lambda r: kcal_from_macros(r[0], r[1], r[2]), axis=1)

    clean = clean.dropna(subset=["kcal_g", "carb_g", "prot_g", "fat_g"]).reset_index(drop=True)
    return clean


@st.cache_data(show_spinner=False)
def _parse_foods_from_bytes(xls_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(BytesIO(xls_bytes))
    if "Todos" in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name="Todos")
        return _normalize_food_df(df)
    dfs = [_normalize_food_df(pd.read_excel(xls, sheet_name=s)) for s in xls.sheet_names]
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def _parse_foods_from_path(path: str, mtime: float) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    if "Todos" in xls.sheet_names:
        return _normalize_food_df(pd.read_excel(xls, sheet_name="Todos"))
    dfs = [_normalize_food_df(pd.read_excel(xls, sheet_name=s)) for s in xls.sheet_names]
    return pd.concat(dfs, ignore_index=True)


def load_foods(uploaded) -> pd.DataFrame:
    try:
        if uploaded is not None:
            data = uploaded.read()
            return _parse_foods_from_bytes(data)
        default_path = "alimentos_800_especificos.xlsx"
        if os.path.exists(default_path):
            return _parse_foods_from_path(default_path, os.path.getmtime(default_path))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        return pd.DataFrame()


# =============================
# NNLS (no SciPy) solver to adjust grams
# =============================

def nnls_iterative(A: np.ndarray, b: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Solve min ||A x - b|| with x>=0 (simple active set) + column scaling.
    - A: (m x n) densities per gram (rows=carb/prot/fat; cols=foods)
    - b: target vector (m,)
    Returns x (n,) in grams per food.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)

    # Column scaling to improve conditioning
    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale == 0] = 1.0
    A_s = A / col_scale

    # LS solution + iterative negative pruning
    x = np.maximum(0.0, np.linalg.lstsq(A_s, b, rcond=None)[0])
    for _ in range(max_iter):
        neg = x < 0
        if not neg.any():
            break
        keep = ~neg
        if keep.sum() == 0:
            return np.zeros_like(x)
        A_sub = A_s[:, keep]
        x_sub = np.maximum(0.0, np.linalg.lstsq(A_sub, b, rcond=None)[0])
        x = np.zeros_like(x)
        x[keep] = x_sub

    # Unscale
    x = np.maximum(0.0, x) / col_scale
    return x


# =============================
# Google OAuth (Authlib) + Supabase helpers
# =============================

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SCOPE = "openid email profile"

# Fallback to your provided URL if not present in secrets
APP_URL = st.secrets.get(
    "APP_URL",
    "https://appdieta-c7oldtsobgapphygatuaodu.streamlit.app/",
)

def supabase_client():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])
    except Exception:
        return None


def login_button():
    """Start Google OIDC flow (Auth Code + PKCE) without relying on session_state for the verifier."""
    # 1) Generate PKCE verifier
    code_verifier = secrets.token_urlsafe(64)

    # 2) Pack it into a signed JWT 'state' so it survives the redirect
    state_key = st.secrets.get("STATE_SECRET", st.secrets.get("GOOGLE_CLIENT_SECRET", "state-dev-key"))
    state_payload = {
        "v": code_verifier,                 # PKCE verifier
        "nonce": secrets.token_urlsafe(16), # CSRF nonce
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,      # 10 minutes
    }
    state_jwt = jwt.encode(state_payload, state_key, algorithm="HS256")

    # 3) Build auth URL with our custom state
    oauth = OAuth2Session(
        client_id=st.secrets.get("GOOGLE_CLIENT_ID", ""),
        scope="openid email profile",
        redirect_uri=APP_URL,
        code_challenge_method="S256",
        code_verifier=code_verifier,
    )
    auth_url, _ = oauth.create_authorization_url(GOOGLE_AUTH_URL, state=state_jwt)

    # 4) Link to Google
    st.link_button("Sign in with Google", auth_url, type="primary")



def handle_oauth_callback():
    """Handle Google redirect: read ?code & ?state, recover PKCE verifier from signed state, exchange token, store user."""
    params = st.query_params

    # If Google returned an explicit error, show it to diagnose
    if "error" in params:
        err = params.get("error")
        if isinstance(err, list): err = err[0]
        desc = params.get("error_description") or params.get("error_subtype") or ""
        if isinstance(desc, list): desc = desc[0]
        st.error(f"Google OAuth error: {err}\n{desc}")
        return None

    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        return None
    if isinstance(code, list):  code = code[0]
    if isinstance(state, list): state = state[0]

    # Decode and validate state (recover PKCE code_verifier)
    try:
        state_key = st.secrets.get("STATE_SECRET", st.secrets.get("GOOGLE_CLIENT_SECRET", "state-dev-key"))
        decoded = jwt.decode(state, state_key, algorithms=["HS256"])  # will check exp
        code_verifier = decoded.get("v", "")
        if not code_verifier:
            st.error("OAuth state did not include PKCE verifier.")
            return None
    except Exception as e:
        st.error(f"Invalid OAuth state: {e}")
        return None

    # Exchange the code for tokens using the recovered verifier
    try:
        oauth = OAuth2Session(
            client_id=st.secrets.get("GOOGLE_CLIENT_ID", ""),
            client_secret=st.secrets.get("GOOGLE_CLIENT_SECRET", ""),
            redirect_uri=APP_URL,
            code_verifier=code_verifier,
        )
        token = oauth.fetch_token(GOOGLE_TOKEN_URL, code=code, include_client_id=True)
        claims = jwt.get_unverified_claims(token["id_token"])
        user = {"sub": claims.get("sub"), "email": claims.get("email")}
        st.session_state["user"] = user
        # Clean URL params
        st.query_params.clear()
        return user
    except Exception as e:
        st.error(f"Failed to complete Google login: {e}")
        return None


def logout():
    for k in ("user", "pkce_verifier", "oauth_state"):
        st.session_state.pop(k, None)


def save_recipe_to_cloud(user_id: str, recipe: dict):
    sb = supabase_client()
    if not sb:
        raise RuntimeError("Supabase is not configured in secrets.")
    row = {
        "user_id": user_id,
        "name": recipe["nombre"],
        "day_type": recipe["tipo_dia"],
        "meal": recipe["comida"],
        "payload": recipe,
    }
    return sb.table("recipes").insert(row).execute()


def load_cloud_recipes(user_id: str):
    sb = supabase_client()
    if not sb:
        return []
    res = sb.table("recipes").select("*").eq("user_id", user_id).order("created_at").execute()
    return res.data or []


# =============================
# Sidebar: Profile & parameters
# =============================

st.sidebar.header("Profile & parameters")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])  # language-agnostic parser supports both
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=65.0, step=0.5)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=178.0, step=0.5)
age = st.sidebar.number_input("Age (years)", min_value=14, max_value=100, value=35, step=1)

st.sidebar.markdown("---")

# Title with tooltip (replaces subheader)
st.sidebar.markdown("""
<div class="tooltip-wrap">
  <h3 style="margin:0">Caloric goal by day type ⓘ</h3>
  <div class="tooltip-content">
    <strong>Activity multipliers:</strong>
    <table>
      <thead>
        <tr><th>Activity level</th><th>Multiplier</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Sedentary</strong></td><td>1.2</td>
            <td>Little or no exercise; desk job.</td></tr>
        <tr><td><strong>Lightly active</strong></td><td>1.375</td>
            <td>Light exercise/sports 1–3 days/week (walking, yoga, easy cycling).</td></tr>
        <tr><td><strong>Moderately active</strong></td><td>1.55</td>
            <td>Moderate exercise/sports 3–5 days/week (gym, running, sports).</td></tr>
        <tr><td><strong>Very active</strong></td><td>1.725</td>
            <td>Hard exercise 6–7 days/week or physically demanding job.</td></tr>
        <tr><td><strong>Extra active</strong></td><td>1.9</td>
            <td>Very intense training (2x/day) or extremely physical job.</td></tr>
      </tbody>
    </table>
  </div>
</div>
""", unsafe_allow_html=True)

cal_mode = st.sidebar.radio("Mode", ["Multiplier", "Manual kcal"], horizontal=True)

if cal_mode == "Multiplier":
    mult_high = st.sidebar.number_input("High", value=1.60, step=0.01, format="%.2f")
    mult_medium = st.sidebar.number_input("Medium", value=1.55, step=0.01, format="%.2f")
    mult_low = st.sidebar.number_input("Low", value=1.50, step=0.01, format="%.2f")
    extra_high = extra_medium = extra_low = 0.0
else:
    st.sidebar.caption("Add/subtract kcal to BMR by day type")
    extra_high = st.sidebar.number_input("Extra kcal - HIGH", value=0, step=10, min_value=-2000, max_value=2000)
    extra_medium = st.sidebar.number_input("Extra kcal - MEDIUM", value=0, step=10, min_value=-2000, max_value=2000)
    extra_low = st.sidebar.number_input("Extra kcal - LOW", value=0, step=10, min_value=-2000, max_value=2000)
    mult_high = mult_medium = mult_low = 1.0

st.sidebar.markdown("---")
st.sidebar.subheader("Daily macros by day type (g/kg of bodyweight)")
st.sidebar.caption("HIGH day")
p_high = st.sidebar.number_input("Protein (g/kg) - HIGH", value=1.4, step=0.1)
g_high = st.sidebar.number_input("Fat (g/kg) - HIGH", value=0.7, step=0.1)
st.sidebar.caption("MEDIUM day")
p_medium = st.sidebar.number_input("Protein (g/kg) - MEDIUM", value=1.7, step=0.1)
g_medium = st.sidebar.number_input("Fat (g/kg) - MEDIUM", value=1.1, step=0.1)
st.sidebar.caption("LOW day")
p_low = st.sidebar.number_input("Protein (g/kg) - LOW", value=2.0, step=0.1)
g_low = st.sidebar.number_input("Fat (g/kg) - LOW", value=1.5, step=0.1)

st.sidebar.markdown("---")
adj_pct = st.sidebar.slider("Total calories adjustment (%)", min_value=-25, max_value=25, value=-10, step=1)

# Auto-calculated carbohydrates in g/kg
st.sidebar.markdown("---")
st.sidebar.subheader("Carbohydrates (g/kg) calculated")

def _tdee_by_method(day_label: str, adj_pct_value: float) -> float:
    bmr_local = mifflin_st_jeor_bmr(sex, weight, height, age)
    if cal_mode == "Multiplier":
        mult_map = {"High": mult_high, "Medium": mult_medium, "Low": mult_low}
        base = bmr_local * mult_map[day_label]
    else:
        extra_map = {"High": float(extra_high), "Medium": float(extra_medium), "Low": float(extra_low)}
        base = bmr_local + extra_map[day_label]
    return base * (1 + adj_pct_value / 100.0)


def carbs_g_per_kg_for_day(day_label: str, p_gkg: float, f_gkg: float, adj_pct_value: float) -> float:
    tdee_x = _tdee_by_method(day_label, adj_pct_value)
    p_day_x = p_gkg * weight
    f_day_x = f_gkg * weight
    c_day_x = max(0.0, (tdee_x - (p_day_x * 4 + f_day_x * 9)) / 4.0)  # g/day
    return round(float(c_day_x / weight), 2)  # g/kg

carbs_high_gkg = carbs_g_per_kg_for_day("High", p_high, g_high, adj_pct)
carbs_medium_gkg = carbs_g_per_kg_for_day("Medium", p_medium, g_medium, adj_pct)
carbs_low_gkg = carbs_g_per_kg_for_day("Low", p_low, g_low, adj_pct)

st.sidebar.caption("HIGH day")
st.sidebar.number_input("Carbohydrates (g/kg) - HIGH", value=carbs_high_gkg, step=0.01, format="%.2f", disabled=True, key="c_high_gkg_ro")
st.sidebar.caption("MEDIUM day")
st.sidebar.number_input("Carbohydrates (g/kg) - MEDIUM", value=carbs_medium_gkg, step=0.01, format="%.2f", disabled=True, key="c_medium_gkg_ro")
st.sidebar.caption("LOW day")
st.sidebar.number_input("Carbohydrates (g/kg) - LOW", value=carbs_low_gkg, step=0.01, format="%.2f", disabled=True, key="c_low_gkg_ro")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload your foods Excel (optional)", type=["xlsx"])

# =============================
# Account (Login / Logout)
# =============================
st.divider()
st.subheader("Account")
user = st.session_state.get("user") or handle_oauth_callback()
if not user:
    st.info("Sign in to save your recipes to the cloud.")
    login_button()
else:
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        st.success(f"Signed in as {user['email']}")
    with col_u2:
        if st.button("Log out"):
            logout()
            _safe_rerun()

# =============================
# Load foods
# =============================
foods = load_foods(uploaded)

# =============================
# Daily calculations (g/day from g/kg and TDEE)
# =============================

bmr = mifflin_st_jeor_bmr(sex, weight, height, age)

# Day type selection and TDEE by mode
day_type = st.selectbox("Day type", ["High", "Medium", "Low"])
if cal_mode == "Multiplier":
    mult_map = {"High": mult_high, "Medium": mult_medium, "Low": mult_low}
    base_tdee = bmr * mult_map[day_type]
else:
    extra_map = {"High": float(extra_high), "Medium": float(extra_medium), "Low": float(extra_low)}
    base_tdee = bmr + extra_map[day_type]

tdee = base_tdee * (1 + adj_pct / 100.0)

# Daily macro targets (g/day)
p_day = {"High": p_high, "Medium": p_medium, "Low": p_low}[day_type] * weight
f_day = {"High": g_high, "Medium": g_medium, "Low": g_low}[day_type] * weight
kcal_from_p_f = p_day * 4 + f_day * 9
c_day = max(0.0, (tdee - kcal_from_p_f) / 4.0)

# --- Top row: metrics + chart aligned ---
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.metric("BMR (kcal/day)", f"{bmr:.0f}")
    st.metric("TDEE (kcal/day)", f"{tdee:.0f}")

with col2:
    st.metric("Protein (g/day)", f"{p_day:.0f}")
    st.metric("Fat (g/day)", f"{f_day:.0f}")
    st.metric("Carbohydrates (g/day)", f"{c_day:.0f}")

with col3:
    st.markdown("### Daily macronutrient split")

    macros_daily_df = pd.DataFrame(
        {
            "Macro": ["Carbohydrates", "Protein", "Fat"],
            "Grams": [c_day, p_day, f_day],
            "kcal": [c_day * 4, p_day * 4, f_day * 9],
        }
    )
    macros_daily_df["% kcal"] = (
        macros_daily_df["kcal"] / macros_daily_df["kcal"].sum() * 100
    ).round(1)

    # Palette
    macro_colors = {
        "Carbohydrates": "#EE9B00",
        "Protein": "#CA6702",
        "Fat": "#BB3E03",
    }

    try:
        pie = (
            alt.Chart(macros_daily_df)
            .mark_arc()
            .encode(
                theta=alt.Theta(field="kcal", type="quantitative"),
                color=alt.Color(
                    field="Macro",
                    type="nominal",
                    scale=alt.Scale(
                        domain=list(macro_colors.keys()),
                        range=list(macro_colors.values()),
                    ),
                ),
                tooltip=["Macro", "Grams", "kcal", "% kcal"],
            )
            .properties(width=360, height=360)
        )
        st.altair_chart(pie, use_container_width=True)
    except Exception:
        st.bar_chart(macros_daily_df.set_index("Macro")["kcal"])


# =============================
# Meal split (editable)
# =============================

st.markdown("### Meal split")
meal_defaults = {
    "Breakfast": {"prot": 0.10, "fat": 0.10, "carb": 0.27},
    "Lunch": {"prot": 0.39, "fat": 0.40, "carb": 0.26},
    "Snack": {"prot": 0.08, "fat": 0.06, "carb": 0.17},
    "Dinner": {"prot": 0.43, "fat": 0.44, "carb": 0.30},
}

with st.expander("Edit split (portion of the day)"):
    for key in meal_defaults:
        st.write(f"**{key}**")
        meal_defaults[key]["prot"] = st.number_input(
            f"Protein ({key})", value=float(meal_defaults[key]["prot"]), step=0.01, format="%.2f", key=f"p_{key}"
        )
        meal_defaults[key]["fat"] = st.number_input(
            f"Fat ({key})", value=float(meal_defaults[key]["fat"]), step=0.01, format="%.2f", key=f"g_{key}"
        )
        meal_defaults[key]["carb"] = st.number_input(
            f"Carbs ({key})", value=float(meal_defaults[key]["carb"]), step=0.01, format="%.2f", key=f"c_{key}"
        )

    totals = {k: sum(meal_defaults[m][k] for m in meal_defaults) for k in ("prot", "fat", "carb")}
    warn_msgs = []
    for k, v in totals.items():
        if not (0.95 <= v <= 1.05):
            warn_msgs.append(f"{k} sum = {v:.2f}")
    if warn_msgs:
        st.warning("; ".join(warn_msgs) + ". Ideally each macro should sum ≈ 1.00 across all meals.")


# =============================
# Per-meal macro targets + Excel (with total row)
# =============================

st.markdown("### Per-meal macro targets")

def meal_targets(meal_name: str, perc: Dict[str, float]) -> Dict[str, float]:
    p_t = p_day * perc["prot"]
    f_t = f_day * perc["fat"]
    c_t = c_day * perc["carb"]
    kcal_t = c_t * 4 + p_t * 4 + f_t * 9
    return {
        "Meal": meal_name,
        "kcal": round(kcal_t, 0),
        "Carbohydrates (g)": round(c_t, 1),
        "Protein (g)": round(p_t, 1),
        "Fat (g)": round(f_t, 1),
    }

meals_summary = pd.DataFrame([
    meal_targets("Breakfast", meal_defaults["Breakfast"]),
    meal_targets("Lunch", meal_defaults["Lunch"]),
    meal_targets("Snack", meal_defaults["Snack"]),
    meal_targets("Dinner", meal_defaults["Dinner"]),
])

# Add TOTAL row
total_row = {
    "Meal": "TOTAL",
    "kcal": round(meals_summary["kcal"].sum(), 0),
    "Carbohydrates (g)": round(meals_summary["Carbohydrates (g)"].sum(), 1),
    "Protein (g)": round(meals_summary["Protein (g)"].sum(), 1),
    "Fat (g)": round(meals_summary["Fat (g)"].sum(), 1),
}
meals_summary_tot = pd.concat([meals_summary, pd.DataFrame([total_row])], ignore_index=True)

# Render HTML table with bold Meal column and bold TOTAL row
def render_meals_table_html(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    html = []
    html.append("""
    <style>
      table.meals-summary {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
      }
      table.meals-summary th, table.meals-summary td {
        border-bottom: 1px solid rgba(0,0,0,.07);
        padding: 6px 8px;
        text-align: right;
      }
      table.meals-summary thead th {
        text-align: left;
        font-weight: 600;
      }
      table.meals-summary th:first-child, table.meals-summary td:first-child {
        text-align: left;
        font-weight: 700; /* Meal column bold */
      }
      table.meals-summary tr:last-child td {
        font-weight: 700; /* TOTAL row bold */
      }
    </style>
    """)
    html.append('<table class="meals-summary">')
    # header
    html.append("<thead><tr>")
    for c in cols:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead>")
    # body
    html.append("<tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                if c == "kcal":
                    cell = f"{val:.0f}"
                else:
                    try:
                        cell = f"{val:.1f}"
                    except Exception:
                        cell = f"{val}"
            else:
                cell = str(val)
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

st.markdown(render_meals_table_html(meals_summary_tot), unsafe_allow_html=True)

# Export Excel
buf_meals = BytesIO()
with pd.ExcelWriter(buf_meals, engine="openpyxl") as writer:
    meals_summary_tot.to_excel(writer, index=False, sheet_name="Per-meal macros")
buf_meals.seek(0)
st.download_button(
    "Download per-meal summary (Excel)",
    data=buf_meals,
    file_name="per_meal_macro_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =============================
# Targets for selected meal
# =============================

st.markdown("### Meal")
meal = st.selectbox("", ["Breakfast", "Lunch", "Snack", "Dinner"], label_visibility="collapsed")
perc = meal_defaults[meal]
pt = p_day * perc["prot"]
ft = f_day * perc["fat"]
ct = c_day * perc["carb"]
kcal_target = ct * 4 + pt * 4 + ft * 9
st.info(f"Target for {meal} → {kcal_target:.0f} kcal | Protein: {pt:.0f} g | Fat: {ft:.0f} g | Carbs: {ct:.0f} g")

# =============================
# Recipe builder
# =============================

st.markdown("### Recipe builder")
if foods.empty:
    st.warning("First upload an Excel with foods (e.g., alimentos_800_especificos.xlsx) or place it in the app folder.")
else:
    df_view = foods.copy().reset_index(drop=True)
    df_view["Marca"] = df_view["Marca"].astype(str).fillna("")

    # Multi-select (max 10) — show brand in parentheses
    df_view["__label__"] = np.where(
        df_view["Marca"].str.strip() != "",
        df_view["Producto"] + " (" + df_view["Marca"] + ")",
        df_view["Producto"]
    )
    options = df_view.index.tolist()
    choices_idx = st.multiselect(
        "Pick up to 10 foods for the recipe",
        options=options,
        format_func=lambda i: df_view.loc[i, "__label__"],
    )
    if len(choices_idx) > 10:
        st.warning("You selected more than 10 items; only the first 10 will be used.")
        choices_idx = choices_idx[:10]

    # Keep one row per product name to avoid ambiguity when later looking up by 'Producto'
    selected = df_view.loc[choices_idx].drop_duplicates("Producto").reset_index(drop=True)

    if not selected.empty:
        # Grams editor with session_state persistence
        editor_key = f"editor_{meal}"
        lock_key = f"{editor_key}_locked"
        current_products = selected["Producto"].tolist()
        prev_products = st.session_state.get(editor_key + "_products")

        if prev_products != current_products:
            base_df = selected[["Producto", "carb_g", "prot_g", "fat_g", "kcal_g"]].copy()
            old_locks = st.session_state.get(lock_key, {})
            locks = {p: bool(old_locks.get(p, False)) for p in base_df["Producto"].tolist()}
            base_df.insert(
                1,
                "Locked",
                pd.Series([locks.get(p, False) for p in base_df["Producto"].tolist()], index=base_df.index)
            )
            base_df.insert(2, "Grams (g)", 0.0)
            st.session_state[editor_key] = base_df
            st.session_state[editor_key + "_products"] = current_products
            st.session_state[lock_key] = locks

        editor_df = st.session_state[editor_key]
        if "Locked" not in editor_df.columns:
            locks = st.session_state.get(lock_key, {p: False for p in editor_df["Producto"].tolist()})
            editor_df.insert(1, "Locked", editor_df["Producto"].map(lambda p: bool(locks.get(p, False))))
            st.session_state[editor_key] = editor_df

        st.write("Enter grams (you can leave 0 and use auto-adjust):")
        editor_df = st.data_editor(
            editor_df,
            key=editor_key + "_table",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Producto": st.column_config.TextColumn("Product", disabled=True),
                "Locked": st.column_config.CheckboxColumn(
                    "Lock 🔒",
                    help="If checked, this ingredient won't be changed by auto-adjust.",
                    default=False,
                ),
                "carb_g": st.column_config.NumberColumn("carb/g", help="Carbs per gram", disabled=True),
                "prot_g": st.column_config.NumberColumn("prot/g", help="Protein per gram", disabled=True),
                "fat_g": st.column_config.NumberColumn("fat/g", help="Fat per gram", disabled=True),
                "kcal_g": st.column_config.NumberColumn("kcal/g", help="kcal per gram", disabled=True),
                "Grams (g)": st.column_config.NumberColumn(step=5.0, min_value=0.0),
            },
        )
        st.session_state[editor_key] = editor_df

        # Sync locks from table (auto-unlock if grams == 0)
        locks = st.session_state.get(lock_key, {})
        for _, r in editor_df.iterrows():
            p = r["Producto"]
            g = float(r["Grams (g)"]) if pd.notna(r["Grams (g)"]) else 0.0
            checked = bool(r.get("Locked", False))
            locks[p] = False if g == 0 else checked
        st.session_state[lock_key] = locks

        # Current totals
        grams = editor_df["Grams (g)"].to_numpy()
        totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
        kcal_tot = float(totals["kcal_g"]) if not np.isnan(totals["kcal_g"]) else 0.0
        carb_tot = float(totals["carb_g"]) if not np.isnan(totals["carb_g"]) else 0.0
        prot_tot = float(totals["prot_g"]) if not np.isnan(totals["prot_g"]) else 0.0
        fat_tot = float(totals["fat_g"]) if not np.isnan(totals["fat_g"]) else 0.0

        colA, colB, colC, colD = st.columns(4)
        colA.metric("kcal", f"{kcal_tot:.0f}", delta=f"{kcal_tot - kcal_target:+.0f}")
        colB.metric("Carbs (g)", f"{carb_tot:.0f}", delta=f"{carb_tot - ct:+.0f}")
        colC.metric("Protein (g)", f"{prot_tot:.0f}", delta=f"{prot_tot - pt:+.0f}")
        colD.metric("Fat (g)", f"{fat_tot:.0f}", delta=f"{fat_tot - ft:+.0f}")

        # Adjust grams
        st.markdown("**Adjust grams**")
        btn_col1, btn_col2 = st.columns([1, 2])

        # 1) Adjust ALL ingredients (match total macro targets)
        with btn_col1:
            if st.button("Adjust ALL (match targets)"):
                A_full = editor_df[["carb_g", "prot_g", "fat_g"]].to_numpy().T  # 3 x N
                b = np.array([ct, pt, ft], dtype=float)
                products = editor_df["Producto"].tolist()
                grams_now = editor_df["Grams (g)"].to_numpy().astype(float)
                locks = st.session_state.get(lock_key, {p: False for p in products})
                locked_idx = [i for i, p in enumerate(products) if locks.get(p, False)]
                unlocked_idx = [i for i, p in enumerate(products) if not locks.get(p, False)]

                if len(unlocked_idx) == 0:
                    st.info("All ingredients are locked (edited manually). Set some to 0 to unlock and adjust.")
                else:
                    # Subtract locked contribution from the target
                    if len(locked_idx) > 0:
                        A_lock = A_full[:, locked_idx]
                        g_lock = grams_now[locked_idx]
                        b_res = b - A_lock @ g_lock
                    else:
                        b_res = b
                    A_un = A_full[:, unlocked_idx]
                    x_un = nnls_iterative(A_un, b_res, max_iter=50)
                    new_grams = grams_now.copy()
                    new_grams[unlocked_idx] = x_un
                    editor_df.loc[:, "Grams (g)"] = new_grams
                    st.session_state[editor_key] = editor_df
                    st.session_state[editor_key + "_programmatic"] = True
                    st.success("Adjusted grams for all unlocked ingredients.")
                    _safe_rerun()

        # 2) Adjust ONLY one selected ingredient
        with btn_col2:
            ing_choice = st.selectbox(
                "Ingredient to adjust (only this one)", editor_df["Producto"].tolist(), key=f"single_sel_{meal}"
            )
            if st.button("Adjust ONLY this ingredient"):
                deficits = np.array([
                    ct - carb_tot,
                    pt - prot_tot,
                    ft - fat_tot,
                ], dtype=float)
                v = (
                    editor_df.loc[editor_df["Producto"] == ing_choice, ["carb_g", "prot_g", "fat_g"]]
                    .to_numpy()
                    .ravel()
                )
                denom = float(np.dot(v, v))
                if denom <= 0 or not np.isfinite(denom):
                    st.warning("Cannot adjust with this ingredient (invalid densities).")
                else:
                    # best 1D LS step; don't allow negative grams
                    g_delta = float(np.dot(v, deficits)) / denom
                    current_g = float(
                        editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"].iloc[0]
                    )
                    new_val = max(0.0, current_g + g_delta)
                    editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"] = new_val
                    st.session_state[editor_key] = editor_df
                    st.session_state[editor_key + "_programmatic"] = True
                    msg = "increased" if g_delta >= 0 else "reduced"
                    st.success(
                        f"Grams {msg} for '{ing_choice}' by {abs(g_delta):.0f} g (new total: {new_val:.0f} g)."
                    )

        # Current recipe detail
        df_curr = editor_df["Producto"].to_frame()
        df_curr["Grams (g)"] = editor_df["Grams (g)"]
        df_curr["Carbohydrates (g)"] = (editor_df["carb_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["Protein (g)"] = (editor_df["prot_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["Fat (g)"] = (editor_df["fat_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["kcal"] = (editor_df["kcal_g"] * editor_df["Grams (g)"]).round(0)

        if not df_curr.empty:
            st.write("**Current recipe detail (before saving)**")
            st.dataframe(df_curr, hide_index=True, use_container_width=True)

        st.markdown("---")
        recipe_name = st.text_input("Recipe name")
        if "recipes" not in st.session_state:
            st.session_state["recipes"] = []

        if st.button("Save recipe"):
            grams = editor_df["Grams (g)"]
            totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
            r = {
                "nombre": recipe_name or f"Recipe {len(st.session_state['recipes']) + 1}",
                "tipo_dia": day_type,
                "comida": meal,
                "objetivo": {"kcal": float(kcal_target), "carb": float(ct), "prot": float(pt), "fat": float(ft)},
                "resultado": {
                    "kcal": float(totals["kcal_g"]),
                    "carb": float(totals["carb_g"]),
                    "prot": float(totals["prot_g"]),
                    "fat": float(totals["fat_g"]),
                },
                "ingredientes": [
                    {"producto": editor_df.loc[i, "Producto"], "gramos": float(editor_df.loc[i, "Grams (g)"])}
                    for i in range(len(editor_df))
                ],
            }
            # Save locally in session
            st.session_state["recipes"].append(r)
            st.success("Recipe saved in the session.")
            # Save to cloud if logged in
            if st.session_state.get("user"):
                try:
                    save_recipe_to_cloud(st.session_state["user"]["sub"], r)
                    st.success("Recipe saved to your account (cloud).")
                except Exception as e:
                    st.warning(f"Cloud save failed: {e}")
            else:
                st.info("Sign in to save this recipe to your account.")

# =============================
# Saved recipes + export
# =============================

st.markdown("## Saved recipes (this session)")
recipes = st.session_state.get("recipes", [])
if not recipes:
    st.caption("No saved recipes yet.")
else:
    for day in ["High", "Medium", "Low"]:
        group = [r for r in recipes if r["tipo_dia"] == day]
        if not group:
            continue
        st.markdown(f"### {day}")
        for r in group:
            with st.expander(f"🍽️ {r['nombre']} · {r['comida']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Target**")
                    st.write(f"{r['objetivo']['kcal']:.0f} kcal | C:{r['objetivo']['carb']:.0f} g · P:{r['objetivo']['prot']:.0f} g · F:{r['objetivo']['fat']:.0f} g")
                with col2:
                    st.write("**Result**")
                    st.write(f"{r['resultado']['kcal']:.0f} kcal | C:{r['resultado']['carb']:.0f} g · P:{r['resultado']['prot']:.0f} g · F:{r['resultado']['fat']:.0f} g")

                # Per-ingredient detail
                ing_rows = []
                for ing in r["ingredientes"]:
                    row = foods[foods["Producto"] == ing["producto"]].head(1)
                    g = float(ing["gramos"])
                    if not row.empty:
                        kcal = float(row["kcal_g"].iloc[0]) * g
                        carb = float(row["carb_g"].iloc[0]) * g
                        prot = float(row["prot_g"].iloc[0]) * g
                        fat = float(row["fat_g"].iloc[0]) * g
                    else:
                        kcal = carb = prot = fat = np.nan
                    ing_rows.append({
                        "Product": ing["producto"],
                        "Grams (g)": round(g, 1),
                        "Carbohydrates (g)": None if pd.isna(carb) else round(carb, 1),
                        "Protein (g)": None if pd.isna(prot) else round(prot, 1),
                        "Fat (g)": None if pd.isna(fat) else round(fat, 1),
                        "kcal": None if pd.isna(kcal) else round(kcal, 0),
                    })
                df_ing = pd.DataFrame(ing_rows)
                st.dataframe(df_ing, hide_index=True, use_container_width=True)

                # Export recipe (Excel)
                buf_recipe = BytesIO()
                with pd.ExcelWriter(buf_recipe, engine="openpyxl") as writer:
                    df_ing.to_excel(writer, index=False, sheet_name=r["nombre"][:31])
                buf_recipe.seek(0)
                st.download_button(
                    "Download recipe (Excel, per-ingredient detail)",
                    data=buf_recipe,
                    file_name=f"{r['nombre'].replace(' ', '_')}_detail.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.caption(
                    f"Recipe totals → kcal: {r['resultado']['kcal']:.0f} · "
                    f"C: {r['resultado']['carb']:.0f} g · "
                    f"P: {r['resultado']['prot']:.0f} g · "
                    f"F: {r['resultado']['fat']:.0f} g"
                )

# Cloud recipes list (optional)
if st.session_state.get("user"):
    st.markdown("## My cloud recipes")
    try:
        cloud = load_cloud_recipes(st.session_state["user"]["sub"])
    except Exception as e:
        cloud = []
        st.warning(f"Couldn't load cloud recipes: {e}")
    if not cloud:
        st.caption("No cloud recipes yet.")
    else:
        cloud_df = pd.DataFrame([
            {
                "Name": r["name"],
                "Day type": r["day_type"],
                "Meal": r["meal"],
                "Created": r["created_at"],
            }
            for r in cloud
        ])
        st.dataframe(cloud_df, use_container_width=True)

# Export ALL session recipes
if st.session_state.get("recipes"):
    st.markdown("\n#### Export ALL recipes (session)")
    buf_all = BytesIO()
    with pd.ExcelWriter(buf_all, engine="openpyxl") as writer:
        summary_rows = []
        for r in st.session_state["recipes"]:
            summary_rows.append({
                "Name": r["nombre"],
                "Day type": r["tipo_dia"],
                "Meal": r["comida"],
                "kcal_target": r["objetivo"]["kcal"],
                "carb_target": r["objetivo"]["carb"],
                "prot_target": r["objetivo"]["prot"],
                "fat_target": r["objetivo"]["fat"],
                "kcal_result": r["resultado"]["kcal"],
                "carb_result": r["resultado"]["carb"],
                "prot_result": r["resultado"]["prot"],
                "fat_result": r["resultado"]["fat"],
            })
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")

        for r in st.session_state["recipes"]:
            ing_rows = []
            for ing in r["ingredientes"]:
                row = foods[foods["Producto"] == ing["producto"]].head(1)
                g = float(ing["gramos"])
                if not row.empty:
                    kcal = float(row["kcal_g"].iloc[0]) * g
                    carb = float(row["carb_g"].iloc[0]) * g
                    prot = float(row["prot_g"].iloc[0]) * g
                    fat = float(row["fat_g"].iloc[0]) * g
                else:
                    kcal = carb = prot = fat = np.nan
                ing_rows.append({
                    "Product": ing["producto"],
                    "Grams (g)": round(g, 1),
                    "Carbohydrates (g)": None if pd.isna(carb) else round(carb, 1),
                    "Protein (g)": None if pd.isna(prot) else round(prot, 1),
                    "Fat (g)": None if pd.isna(fat) else round(fat, 1),
                    "kcal": None if pd.isna(kcal) else round(kcal, 0),
                })
            pd.DataFrame(ing_rows).to_excel(writer, index=False, sheet_name=r["nombre"][:31])

    buf_all.seek(0)
    st.download_button(
        "Download ALL recipes (Excel)",
        data=buf_all,
        file_name="session_recipes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown(
    """
    ---
    **Note**: Data and recipes are stored only for this browser session (and in the cloud if you're signed in).
    To enable cloud saving, set the following in **Settings → Secrets**:

    - APP_URL = the public URL of your app
    - GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
    - SUPABASE_URL / SUPABASE_ANON_KEY

    In Supabase, create a table `recipes` with columns: id (uuid, default gen_random_uuid()),
    user_id (text), name (text), day_type (text), meal (text), payload (jsonb), created_at (timestamptz default now()).
    Keep RLS disabled for this table if you're using Google OIDC directly; we filter by user_id in queries.
    """
)
