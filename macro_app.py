import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
from io import BytesIO
from typing import Dict, List, Optional
import altair as alt

# ==== Utilities ====

def _safe_rerun():
    """Streamlit compatibility rerun helper."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _norm_txt(s: str) -> str:
    """Normalize text: lowercase, strip, remove accents, collapse spaces."""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

def _to_float_series(x: pd.Series) -> pd.Series:
    """Robustly convert a Series to float, handling commas/spaces."""
    if x.dtype.kind in {"i", "u", "f"}:
        return x.astype(float)
    return pd.to_numeric(
        x.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )

def kcal_from_macros(carb_g: float, prot_g: float, fat_g: float) -> float:
    """Calculate kcal from macros."""
    return carb_g * 4 + prot_g * 4 + fat_g * 9

def harris_benedict_revised_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    """Calculate BMR using revised Harris-Benedict formula (Roza & Shizgal, 1984)."""
    sex_norm = _norm_txt(sex)
    male_tokens = {"hombre", "masculino", "varon", "varón", "male", "man"}
    if sex_norm in male_tokens:
        return 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    return 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)

# ==== Food Data Normalization ====

EXPECTED_NAME_KEYS = ["producto", "alimento", "nombre"]
EXPECTED_BRAND_KEYS = ["marca"]
EXPECTED_CAT_KEYS = ["categoria", "categoría"]
EXPECTED_SUBCAT_KEYS = ["subcategoria", "subcategoría"]
KCAL_PER_G_KEYS = [
    "energia (kcal/g)", "energia kcal/g", "calorias (kcal/g)",
    "calorias kcal/g", "kcal/g"
]
KCAL_PER_100G_KEYS = [
    "energia (kcal/100g)", "energia kcal/100g", "calorias (kcal/100g)",
    "calorias kcal/100g", "kcal/100g", "kcal"
]
CARB_PER_G_KEYS = ["carbohidratos (g/g)", "hidratos (g/g)", "carbs (g/g)", "ch/g"]
CARB_PER_100G_KEYS = ["carbohidratos (g/100g)", "hidratos (g/100g)", "carbs (g/100g)", "ch"]
PROT_PER_G_KEYS = ["proteinas (g/g)", "proteínas (g/g)", "protein (g/g)", "p/g"]
PROT_PER_100G_KEYS = ["proteinas (g/100g)", "proteínas (g/100g)", "protein (g/100g)", "p"]
FAT_PER_G_KEYS = ["grasas (g/g)", "lipidos (g/g)", "g/g"]
FAT_PER_100G_KEYS = ["grasas (g/100g)", "lipidos (g/100g)", "g"]

def _find_first(cols_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    """Find first column in cols_map matching any of candidates."""
    for key in candidates:
        if key in cols_map:
            return cols_map[key]
    return None

def _normalize_food_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw food dataframe columns, unify macro units and names."""
    cols_map = {_norm_txt(c): c for c in df.columns}
    name_col = _find_first(cols_map, EXPECTED_NAME_KEYS)
    if not name_col:
        raise ValueError("Couldn't find a product name column (e.g., 'Producto'/'Alimento').")
    brand_col = _find_first(cols_map, EXPECTED_BRAND_KEYS)
    cat_col = _find_first(cols_map, EXPECTED_CAT_KEYS)
    subcat_col = _find_first(cols_map, EXPECTED_SUBCAT_KEYS)

    kcal_g = None
    kcal_g_src = _find_first(cols_map, KCAL_PER_G_KEYS)
    kcal_100g_src = _find_first(cols_map, KCAL_PER_100G_KEYS)
    if kcal_g_src:
        kcal_g = _to_float_series(df[kcal_g_src])
    elif kcal_100g_src:
        kcal_g = _to_float_series(df[kcal_100g_src]) / 100.0

    carb_g = None
    carb_g_src = _find_first(cols_map, CARB_PER_G_KEYS)
    carb_100g_src = _find_first(cols_map, CARB_PER_100G_KEYS)
    if carb_g_src:
        carb_g = _to_float_series(df[carb_g_src])
    elif carb_100g_src:
        carb_g = _to_float_series(df[carb_100g_src]) / 100.0

    prot_g = None
    prot_g_src = _find_first(cols_map, PROT_PER_G_KEYS)
    prot_100g_src = _find_first(cols_map, PROT_PER_100G_KEYS)
    if prot_g_src:
        prot_g = _to_float_series(df[prot_g_src])
    elif prot_100g_src:
        prot_g = _to_float_series(df[prot_100g_src]) / 100.0

    fat_g = None
    fat_g_src = _find_first(cols_map, FAT_PER_G_KEYS)
    fat_100g_src = _find_first(cols_map, FAT_PER_100G_KEYS)
    if fat_g_src:
        fat_g = _to_float_series(df[fat_g_src])
    elif fat_100g_src:
        fat_g = _to_float_series(df[fat_100g_src]) / 100.0

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

    # Compute missing kcal_g from macros if possible
    missing_kcal_mask = clean["kcal_g"].isna()
    macros_available = clean[["carb_g", "prot_g", "fat_g"]].notna().all(axis=1)
    compute_mask = missing_kcal_mask & macros_available
    clean.loc[compute_mask, "kcal_g"] = clean.loc[compute_mask, ["carb_g", "prot_g", "fat_g"]].apply(
        lambda r: kcal_from_macros(r.carb_g, r.prot_g, r.fat_g), axis=1
    )
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
def _parse_foods_from_path(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return _normalize_food_df(pd.read_csv(path))
    xls = pd.ExcelFile(path)
    if "Todos" in xls.sheet_names:
        return _normalize_food_df(pd.read_excel(xls, sheet_name="Todos"))
    dfs = [_normalize_food_df(pd.read_excel(xls, sheet_name=s)) for s in xls.sheet_names]
    return pd.concat(dfs, ignore_index=True)

def load_foods(file_path: str) -> pd.DataFrame:
    """Load foods from a local CSV/XLSX path."""
    try:
        if not file_path:
            return pd.DataFrame()
        if os.path.exists(file_path):
            return _parse_foods_from_path(file_path)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"No se pudo leer el archivo de alimentos ({file_path}): {e}")
        return pd.DataFrame()

# ==== NNLS Solver without SciPy ====

def nnls_iterative(A: np.ndarray, b: np.ndarray, max_iter=50) -> np.ndarray:
    """
    Solve min ||Ax - b|| with x >= 0 using simple iterative pruning approach.
    A shape: (m x n), b shape: (m,)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)

    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale == 0] = 1.0
    A_scaled = A / col_scale

    x = np.maximum(0.0, np.linalg.lstsq(A_scaled, b, rcond=None)[0])

    for _ in range(max_iter):
        neg = x < 0
        if not neg.any():
            break
        keep = ~neg
        if keep.sum() == 0:
            return np.zeros_like(x)
        A_sub = A_scaled[:, keep]
        x_sub = np.maximum(0.0, np.linalg.lstsq(A_sub, b, rcond=None)[0])
        x = np.zeros_like(x)
        x[keep] = x_sub

    x = np.maximum(0.0, x) / col_scale
    return x


# ==== Streamlit App UI ====

st.set_page_config(page_title="DIET APP · Meal Planner", layout="wide")

st.markdown("""
<style>
:root { --bg: #f7f9fc; --card:#ffffff; --text:#0f172a; --muted:#64748b; --accent:#2563eb; }
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.01em; }
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 12px;
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.title("AppDieta · Planificador")
st.caption("Interfaz simplificada y enfoque en objetivos diarios + diseño de recetas.")

# Sidebar: Profile and parameters

st.sidebar.header("Profile & parameters")
sex = st.sidebar.selectbox("Sexo", ["Hombre", "Mujer"])
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=65.0, step=0.5)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=178.0, step=0.5)
age = st.sidebar.number_input("Age (years)", min_value=14, max_value=100, value=35, step=1)

st.sidebar.markdown("---")
bmr_mode = st.sidebar.radio("Modo kcal basal", ["Automático (Harris-Benedict revisada)", "Manual"], horizontal=False)
bmr_manual = None
if bmr_mode == "Manual":
    bmr_manual = st.sidebar.number_input("Kcal basal manual", min_value=800.0, max_value=5000.0, value=1800.0, step=10.0)

st.sidebar.markdown("---")
cal_mode = st.sidebar.radio("Modo gasto diario", ["Multiplicador", "Kcal extra manual"], horizontal=True)

# Multipliers or manual extras input
if cal_mode == "Multiplicador":
    mult_high = st.sidebar.number_input("Multiplicador - Alta actividad", value=1.60, step=0.01, format="%.2f")
    mult_medium = st.sidebar.number_input("Multiplicador - Media actividad", value=1.55, step=0.01, format="%.2f")
    mult_low = st.sidebar.number_input("Multiplicador - Baja actividad", value=1.50, step=0.01, format="%.2f")
    extra_high = extra_medium = extra_low = 0.0
else:
    st.sidebar.caption("Añadir o quitar kcal a la basal según tipo de día")
    extra_high = st.sidebar.number_input("Kcal extra - Día ALTO", value=0, step=10, min_value=-2000, max_value=2000)
    extra_medium = st.sidebar.number_input("Kcal extra - Día MEDIO", value=0, step=10, min_value=-2000, max_value=2000)
    extra_low = st.sidebar.number_input("Kcal extra - Día BAJO", value=0, step=10, min_value=-2000, max_value=2000)
    mult_high = mult_medium = mult_low = 1.0

st.sidebar.markdown("---")
st.sidebar.subheader("Daily macros by day type (g / kg bodyweight)")

# Protein and fat inputs per day type
def input_macro(label: str, default: float) -> float:
    return st.sidebar.number_input(label, value=default, step=0.1, format="%.2f")

st.sidebar.caption("HIGH activity day")
p_high = input_macro("Protein (g/kg) - HIGH", 1.4)
g_high = input_macro("Fat (g/kg) - HIGH", 0.7)

st.sidebar.caption("MEDIUM activity day")
p_medium = input_macro("Protein (g/kg) - MEDIUM", 1.7)
g_medium = input_macro("Fat (g/kg) - MEDIUM", 1.1)

st.sidebar.caption("LOW activity day")
p_low = input_macro("Protein (g/kg) - LOW", 2.0)
g_low = input_macro("Fat (g/kg) - LOW", 1.5)

st.sidebar.markdown("---")
adj_pct = st.sidebar.slider("Ajuste objetivo kcal (%)", min_value=-20, max_value=20, value=0, step=1)

# Calculate carbohydrates automatically based on kcal target and protein/fat kcal
def calculate_tdee_and_macros(
    sex: str,
    weight: float,
    height: float,
    age: int,
    day_type: str,
    cal_mode: str,
    mults: Dict[str, float],
    extras: Dict[str, float],
    adj_pct: float,
    p_gkg: float,
    f_gkg: float,
    bmr_mode: str,
    bmr_manual: Optional[float],
) -> Dict[str, float]:
    bmr_val = float(bmr_manual) if bmr_mode == "Manual" and bmr_manual is not None else harris_benedict_revised_bmr(sex, weight, height, age)

    # Multipliers always have priority over manual extra mode
    multiplier_val = mults.get(day_type, 0.0)
    if multiplier_val > 0:
        tdee_base = bmr_val * multiplier_val
        kcal_source = "multiplicador"
    elif cal_mode == "Kcal extra manual":
        tdee_base = bmr_val + extras.get(day_type, 0.0)
        kcal_source = "kcal extra"
    else:
        tdee_base = bmr_val
        kcal_source = "basal"

    tdee_adjusted = tdee_base * (1 + adj_pct / 100)
    p_day = p_gkg * weight
    f_day = f_gkg * weight
    kcal_pf = p_day * 4 + f_day * 9
    c_day = (tdee_adjusted - kcal_pf) / 4

    carb_negative = c_day < 0
    if carb_negative:
        c_day = 0.0

    return {
        "bmr": bmr_val,
        "tdee": tdee_adjusted,
        "protein_g": p_day,
        "fat_g": f_day,
        "carb_g": c_day,
        "carb_negative": carb_negative,
        "kcal_source": kcal_source,
    }

mults_dict = {"High": mult_high, "Medium": mult_medium, "Low": mult_low}
extras_dict = {"High": extra_high, "Medium": extra_medium, "Low": extra_low}

carbs_high_gkg = calculate_tdee_and_macros(sex, weight, height, age, "High", cal_mode, mults_dict, extras_dict, adj_pct, p_high, g_high, bmr_mode, bmr_manual)["carb_g"] / weight
carbs_medium_gkg = calculate_tdee_and_macros(sex, weight, height, age, "Medium", cal_mode, mults_dict, extras_dict, adj_pct, p_medium, g_medium, bmr_mode, bmr_manual)["carb_g"] / weight
carbs_low_gkg = calculate_tdee_and_macros(sex, weight, height, age, "Low", cal_mode, mults_dict, extras_dict, adj_pct, p_low, g_low, bmr_mode, bmr_manual)["carb_g"] / weight

st.sidebar.caption("Carbohidratos (g/kg) calculados automáticamente")
st.sidebar.number_input("CH (g/kg) - ALTO", value=round(carbs_high_gkg, 2), step=0.01, disabled=True)
st.sidebar.number_input("CH (g/kg) - MEDIO", value=round(carbs_medium_gkg, 2), step=0.01, disabled=True)
st.sidebar.number_input("CH (g/kg) - BAJO", value=round(carbs_low_gkg, 2), step=0.01, disabled=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Fuente de alimentos")
foods_path = st.sidebar.text_input("Ruta local CSV/XLSX", value="alimentos.csv")
st.sidebar.caption("Pon tu archivo en esa ruta (ej: alimentos.csv o data/alimentos.csv).")

# ==== Load foods ====

foods = load_foods(foods_path)

# ==== Main app calculations ====

bmr = float(bmr_manual) if bmr_mode == "Manual" and bmr_manual is not None else harris_benedict_revised_bmr(sex, weight, height, age)

day_type = st.selectbox("Selecciona tipo de día", ["High", "Medium", "Low"], format_func=lambda x: {"High":"Alta actividad","Medium":"Media actividad","Low":"Baja actividad"}[x])

macros = calculate_tdee_and_macros(
    sex, weight, height, age, day_type,
    cal_mode, mults_dict, extras_dict, adj_pct,
    {"High": p_high, "Medium": p_medium, "Low": p_low}[day_type],
    {"High": g_high, "Medium": g_medium, "Low": g_low}[day_type],
    bmr_mode, bmr_manual
)

tdee = macros["tdee"]
p_day = macros["protein_g"]
f_day = macros["fat_g"]
c_day = macros["carb_g"]

if macros.get("carb_negative"):
    st.warning("Las kcal objetivo no alcanzan para proteína+grasa configuradas. Se fijan CH en 0 g.")

# Display metrics and chart

col1, col2, col3 = st.columns([1,1,1.2])

with col1:
    st.metric("Kcal basal (kcal/día)", f"{bmr:.0f}")
    st.metric("Objetivo kcal día (kcal/día)", f"{tdee:.0f}")

with col2:
    st.metric("Proteína (g/día)", f"{p_day:.0f}")
    st.metric("Grasa (g/día)", f"{f_day:.0f}")
    st.metric("CH (g/día)", f"{c_day:.0f}")

with col3:
    st.markdown("### Daily macronutrient split")

    macros_df = pd.DataFrame({
        "Macro": ["Carbohydrates", "Protein", "Fat"],
        "Grams": [c_day, p_day, f_day],
        "kcal": [c_day*4, p_day*4, f_day*9]
    })
    macros_df["% kcal"] = (macros_df["kcal"] / macros_df["kcal"].sum() * 100).round(1)

    macro_colors = {
        "Carbohydrates": "#EE9B00",
        "Protein": "#CA6702",
        "Fat": "#BB3E03"
    }

    try:
        pie = (
            alt.Chart(macros_df)
            .mark_arc()
            .encode(
                theta=alt.Theta(field="kcal", type="quantitative"),
                color=alt.Color(field="Macro", type="nominal",
                                scale=alt.Scale(domain=list(macro_colors.keys()), range=list(macro_colors.values()))),
                tooltip=["Macro", "Grams", "kcal", "% kcal"]
            )
            .properties(width=360, height=360)
        )
        st.altair_chart(pie, use_container_width=True)
    except Exception:
        st.bar_chart(macros_df.set_index("Macro")["kcal"])

# ==== Meal split with editable portions ====

st.markdown("### Meal split")

meal_defaults = {
    "Breakfast": {"prot": 0.10, "fat": 0.10, "carb": 0.27},
    "Lunch": {"prot": 0.39, "fat": 0.40, "carb": 0.26},
    "Snack": {"prot": 0.08, "fat": 0.06, "carb": 0.17},
    "Dinner": {"prot": 0.43, "fat": 0.44, "carb": 0.30},
}

with st.expander("Edit split (portion of daily macros)"):
    for meal in meal_defaults:
        st.write(f"**{meal}**")
        for macro in ["prot", "fat", "carb"]:
            new_val = st.number_input(f"{macro.capitalize()} fraction ({meal})",
                                      value=float(meal_defaults[meal][macro]), step=0.01,
                                      format="%.2f", key=f"{macro}_{meal}")
            meal_defaults[meal][macro] = new_val

    totals = {macro: sum(meal_defaults[m][macro] for m in meal_defaults) for macro in ["prot", "fat", "carb"]}

    warn_msgs = [f"{k} sum = {v:.2f}" for k,v in totals.items() if not 0.95 <= v <= 1.05]

    if warn_msgs:
        st.warning("; ".join(warn_msgs) + ". Ideally each macro sum should be near 1.00.")

# ==== Per meal macro targets table and Excel export ====

st.markdown("### Per-meal macro targets")

def meal_targets(meal_name: str, perc: Dict[str, float]) -> Dict[str, float]:
    p = p_day * perc["prot"]
    f = f_day * perc["fat"]
    c = c_day * perc["carb"]
    kcal = c*4 + p*4 + f*9
    return {
        "Meal": meal_name,
        "kcal": round(kcal, 0),
        "Carbohydrates (g)": round(c, 1),
        "Protein (g)": round(p, 1),
        "Fat (g)": round(f, 1)
    }

meals_summary = pd.DataFrame([
    meal_targets(meal, meal_defaults[meal]) for meal in meal_defaults
])

total_row = {
    "Meal": "TOTAL",
    "kcal": meals_summary["kcal"].sum(),
    "Carbohydrates (g)": meals_summary["Carbohydrates (g)"].sum(),
    "Protein (g)": meals_summary["Protein (g)"].sum(),
    "Fat (g)": meals_summary["Fat (g)"].sum()
}

meals_summary_tot = pd.concat([meals_summary, pd.DataFrame([total_row])], ignore_index=True)

def render_meals_table_html(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    html = ["""
    <style>
    table.meals-summary {width: 100%; border-collapse: collapse; font-size: 0.95rem;}
    table.meals-summary th, td {padding: 6px 8px; border-bottom: 1px solid rgba(0,0,0,.07); text-align: right;}
    table.meals-summary thead th {text-align: left; font-weight: 600;}
    table.meals-summary th:first-child, td:first-child {text-align: left; font-weight: 700;}
    table.meals-summary tr:last-child td {font-weight: 700;}
    </style>
    """, '<table class="meals-summary"><thead><tr>']

    for c in cols:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html.append("<tr>")
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cell = f"{val:.0f}" if c=="kcal" else f"{val:.1f}"
            else:
                cell = str(val)
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

st.markdown(render_meals_table_html(meals_summary_tot), unsafe_allow_html=True)

st.download_button(
    "Descargar resumen de macros por comida (CSV)",
    data=meals_summary_tot.to_csv(index=False).encode("utf-8"),
    file_name="resumen_macros_comidas.csv",
    mime="text/csv"
)

# ==== Targets for selected meal ====

st.markdown("### Meal")

meal = st.selectbox("Select a meal", ["Breakfast", "Lunch", "Snack", "Dinner"], label_visibility="collapsed")
perc = meal_defaults[meal]

pt = p_day * perc["prot"]
ft = f_day * perc["fat"]
ct = c_day * perc["carb"]
kcal_target = ct*4 + pt*4 + ft*9

st.info(
    f"Target for {meal} → {kcal_target:.0f} kcal | Protein: {pt:.0f} g | Fat: {ft:.0f} g | Carbs: {ct:.0f} g"
)

# ==== Recipe builder ====

st.markdown("### Recipe builder")

if foods.empty:
    st.warning("No se encontró el archivo de alimentos. Coloca tu CSV/XLSX en la ruta indicada en la barra lateral.")
else:
    df_view = foods.copy().reset_index(drop=True)
    df_view["Marca"] = df_view["Marca"].fillna("").astype(str)

    df_view["__label__"] = np.where(
        df_view["Marca"].str.strip() != "", 
        df_view["Producto"] + " (" + df_view["Marca"] + ")",
        df_view["Producto"]
    )

    options = df_view.index.tolist()
    choices_idx = st.multiselect(
        "Select up to 10 foods for the recipe",
        options=options,
        format_func=lambda i: df_view.loc[i, "__label__"],
        default=[]
    )

    if len(choices_idx) > 10:
        st.warning("You selected more than 10 items; only the first 10 will be used.")
        choices_idx = choices_idx[:10]

    selected = df_view.loc[choices_idx].drop_duplicates("Producto").reset_index(drop=True)

    if not selected.empty:
        editor_key = f"recipe_editor_{meal.lower()}"
        lock_key = f"{editor_key}_locked"

        current_products = selected["Producto"].tolist()
        prev_products = st.session_state.get(editor_key + "_products")

        if prev_products != current_products:
            base_df = selected[["Producto", "carb_g", "prot_g", "fat_g", "kcal_g"]].copy()
            old_locks = st.session_state.get(lock_key, {})
            locks = {p: bool(old_locks.get(p, False)) for p in base_df["Producto"].tolist()}
            base_df.insert(1, "Locked", pd.Series([locks.get(p, False) for p in base_df["Producto"]], index=base_df.index))
            base_df.insert(2, "Grams (g)", 0.0)
            st.session_state[editor_key] = base_df
            st.session_state[editor_key + "_products"] = current_products
            st.session_state[lock_key] = locks

        editor_df = st.session_state[editor_key]

        if "Locked" not in editor_df.columns:
            locks = st.session_state.get(lock_key, {p: False for p in editor_df["Producto"]})
            editor_df.insert(1, "Locked", editor_df["Producto"].map(lambda p: bool(locks.get(p, False))))
            st.session_state[editor_key] = editor_df

        st.write("Introduce gramos por ingrediente (enteros; usa 0 para desbloquear).")

        editor_df = st.data_editor(
            editor_df,
            key=editor_key + "_table",
            use_container_width=True,
            hide_index=True,
            column_config = {
                "Producto": st.column_config.TextColumn("Product", disabled=True),
                "Locked": st.column_config.CheckboxColumn("Lock 🔒", help="Lock ingredient grams from auto-adjust.", default=False),
                "carb_g": st.column_config.NumberColumn("Carb/g", disabled=True, format="%.3f"),
                "prot_g": st.column_config.NumberColumn("Prot/g", disabled=True, format="%.3f"),
                "fat_g": st.column_config.NumberColumn("Fat/g", disabled=True, format="%.3f"),
                "kcal_g": st.column_config.NumberColumn("kcal/g", disabled=True, format="%.3f"),
                "Grams (g)": st.column_config.NumberColumn(step=1.0, min_value=0.0, format="%d"),
            }
        )

        st.session_state[editor_key] = editor_df

        # Sync locks from table, unlocking grams=0
        locks = st.session_state.get(lock_key, {})
        for _, row in editor_df.iterrows():
            p = row["Producto"]
            g = float(row.get("Grams (g)", 0) or 0)
            checked = bool(row.get("Locked", False))
            locks[p] = False if g == 0 else checked
        st.session_state[lock_key] = locks

        # Compute totals for recipe macros and kcal
        grams = editor_df["Grams (g)"].to_numpy(dtype=float)
        totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
        kcal_tot = float(totals["kcal_g"])
        carb_tot = float(totals["carb_g"])
        prot_tot = float(totals["prot_g"])
        fat_tot = float(totals["fat_g"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("kcal", f"{kcal_tot:.0f}", delta=f"{kcal_tot - kcal_target:+.0f}")
        c2.metric("Carbs (g)", f"{carb_tot:.0f}", delta=f"{carb_tot - ct:+.0f}")
        c3.metric("Protein (g)", f"{prot_tot:.0f}", delta=f"{prot_tot - pt:+.0f}")
        c4.metric("Fat (g)", f"{fat_tot:.0f}", delta=f"{fat_tot - ft:+.0f}")

        dev_kcal = abs(kcal_tot - kcal_target) / max(kcal_target, 1) * 100
        dev_carb = abs(carb_tot - ct) / max(ct, 1) * 100
        dev_prot = abs(prot_tot - pt) / max(pt, 1) * 100
        dev_fat = abs(fat_tot - ft) / max(ft, 1) * 100
        max_dev = max(dev_kcal, dev_carb, dev_prot, dev_fat)

        st.caption(
            f"Desviación objetivo → kcal: {dev_kcal:.1f}% · CH: {dev_carb:.1f}% · P: {dev_prot:.1f}% · G: {dev_fat:.1f}%"
        )
        if max_dev <= 2:
            st.success("La receta está dentro del umbral de ±2% del objetivo.")
        else:
            st.warning("No se alcanza el objetivo ±2%. Sugerencia: añade o elimina alimentos para mejorar el ajuste.")

        st.markdown("**Ajuste de gramos**")

        btn_col1, btn_col2 = st.columns([1, 2])

        # Adjust all unlocked ingredients to meet targets
        with btn_col1:
            if st.button("Calcular gramos automáticamente"):
                A_full = editor_df[["carb_g", "prot_g", "fat_g"]].to_numpy().T  # shape 3 x n
                b_vec = np.array([ct, pt, ft], dtype=float)
                products = editor_df["Producto"].tolist()
                grams_now = editor_df["Grams (g)"].to_numpy(dtype=float)
                locks = st.session_state.get(lock_key, {p: False for p in products})
                locked_idx = [i for i, p in enumerate(products) if locks.get(p, False)]
                unlocked_idx = [i for i, p in enumerate(products) if not locks.get(p, False)]

                if not unlocked_idx:
                    st.info("All ingredients are locked. Set some grams to 0 to unlock for adjustment.")
                else:
                    if locked_idx:
                        A_lock = A_full[:, locked_idx]
                        g_lock = grams_now[locked_idx]
                        b_res = b_vec - A_lock @ g_lock
                    else:
                        b_res = b_vec
                    A_un = A_full[:, unlocked_idx]
                    x_un = nnls_iterative(A_un, b_res)
                    new_grams = grams_now.copy()
                    new_grams[unlocked_idx] = np.rint(x_un)
                    editor_df.loc[:, "Grams (g)"] = new_grams.astype(int)
                    st.session_state[editor_key] = editor_df
                    st.success("Gramos calculados para ingredientes desbloqueados.")
                    _safe_rerun()

        # Adjust grams of a single selected ingredient
        with btn_col2:
            ing_choice = st.selectbox(
                "Ingredient to adjust only", editor_df["Producto"].tolist(),
                key=f"single_sel_{meal}"
            )
            if st.button("Ajustar solo este ingrediente"):
                deficits = np.array([ct - carb_tot, pt - prot_tot, ft - fat_tot], dtype=float)
                v = editor_df.loc[editor_df["Producto"] == ing_choice, ["carb_g", "prot_g", "fat_g"]].to_numpy().ravel()
                denom = float(np.dot(v, v))
                if denom <= 0 or not np.isfinite(denom):
                    st.warning("Cannot adjust with this ingredient (invalid macro densities).")
                else:
                    g_delta = float(np.dot(v, deficits)) / denom
                    current_g = float(editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"].iloc[0])
                    new_val = int(max(0.0, round(current_g + g_delta)))
                    editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"] = new_val
                    st.session_state[editor_key] = editor_df
                    msg = "aumentados" if g_delta >= 0 else "reducidos"
                    st.success(f"Gramos {msg} en '{ing_choice}' ({abs(g_delta):.0f} g; nuevo total: {new_val} g).")

        # Display current recipe macro breakdown
        df_curr = editor_df[["Producto", "Grams (g)"]].copy()
        df_curr["Carbohydrates (g)"] = (editor_df["carb_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["Protein (g)"] = (editor_df["prot_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["Fat (g)"] = (editor_df["fat_g"] * editor_df["Grams (g)"]).round(1)
        df_curr["kcal"] = (editor_df["kcal_g"] * editor_df["Grams (g)"]).round()

        if not df_curr.empty:
            st.write("**Current recipe detail (before saving)**")
            st.dataframe(df_curr, hide_index=True, use_container_width=True)

        st.markdown("---")

        recipe_name = st.text_input("Recipe name", key="recipe_name_input")

        if "recipes" not in st.session_state:
            st.session_state["recipes"] = []

        if st.button("Save recipe"):
            grams = editor_df["Grams (g)"]
            totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
            recipe_dict = {
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
            st.session_state["recipes"].append(recipe_dict)
            st.success("Recipe saved in this session.")

# ==== Saved Recipes and Export ====

st.markdown("## Saved recipes (this session)")

recipes = st.session_state.get("recipes", [])

if not recipes:
    st.caption("Aún no hay recetas guardadas.")
else:
    for day in ["High", "Medium", "Low"]:
        group = [r for r in recipes if r["tipo_dia"] == day]
        if group:
            st.markdown(f"### {day} day recipes")
            for r in group:
                with st.expander(f"🍽️ {r['nombre']} · {r['comida']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Target macros**")
                        st.write(f"{r['objetivo']['kcal']:.0f} kcal | C:{r['objetivo']['carb']:.0f} g · P:{r['objetivo']['prot']:.0f} g · F:{r['objetivo']['fat']:.0f} g")
                    with col2:
                        st.write("**Achieved macros**")
                        st.write(f"{r['resultado']['kcal']:.0f} kcal | C:{r['resultado']['carb']:.0f} g · P:{r['resultado']['prot']:.0f} g · F:{r['resultado']['fat']:.0f} g")

                    ing_rows = []
                    for ing in r["ingredientes"]:
                        row = foods[foods["Producto"] == ing["producto"]].head(1)
                        grams = float(ing["gramos"])
                        if not row.empty:
                            kcal_val = row["kcal_g"].iloc[0] * grams
                            carb_val = row["carb_g"].iloc[0] * grams
                            prot_val = row["prot_g"].iloc[0] * grams
                            fat_val = row["fat_g"].iloc[0] * grams
                        else:
                            kcal_val = carb_val = prot_val = fat_val = np.nan
                        ing_rows.append({
                            "Product": ing["producto"],
                            "Grams (g)": int(round(grams, 0)),
                            "Carbohydrates (g)": round(carb_val, 1) if not pd.isna(carb_val) else None,
                            "Protein (g)": round(prot_val, 1) if not pd.isna(prot_val) else None,
                            "Fat (g)": round(fat_val, 1) if not pd.isna(fat_val) else None,
                            "kcal": round(kcal_val, 0) if not pd.isna(kcal_val) else None,
                        })

                    df_ing = pd.DataFrame(ing_rows)
                    st.dataframe(df_ing, hide_index=True, use_container_width=True)

                    st.download_button(
                        "Descargar receta (CSV)",
                        data=df_ing.to_csv(index=False).encode("utf-8"),
                        file_name=f"{r['nombre'].replace(' ', '_')}_detail.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        f"Recipe totals → kcal: {r['resultado']['kcal']:.0f} · "
                        f"C: {r['resultado']['carb']:.0f} g · "
                        f"P: {r['resultado']['prot']:.0f} g · "
                        f"F: {r['resultado']['fat']:.0f} g"
                    )

    st.markdown("#### Exportar TODAS las recetas")

    summary_rows = []
    detail_rows = []
    for r in recipes:
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

        for ing in r["ingredientes"]:
            row = foods[foods["Producto"] == ing["producto"]].head(1)
            grams = float(ing["gramos"])
            if not row.empty:
                kcal_val = row["kcal_g"].iloc[0] * grams
                carb_val = row["carb_g"].iloc[0] * grams
                prot_val = row["prot_g"].iloc[0] * grams
                fat_val = row["fat_g"].iloc[0] * grams
            else:
                kcal_val = carb_val = prot_val = fat_val = np.nan
            detail_rows.append({
                "recipe": r["nombre"],
                "day_type": r["tipo_dia"],
                "meal": r["comida"],
                "product": ing["producto"],
                "grams": int(round(grams, 0)),
                "carb_g": round(carb_val, 1) if not pd.isna(carb_val) else None,
                "prot_g": round(prot_val, 1) if not pd.isna(prot_val) else None,
                "fat_g": round(fat_val, 1) if not pd.isna(fat_val) else None,
                "kcal": round(kcal_val, 0) if not pd.isna(kcal_val) else None,
            })

    st.download_button(
        "Descargar resumen recetas (CSV)",
        data=pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8"),
        file_name="resumen_recetas.csv",
        mime="text/csv"
    )

    st.download_button(
        "Descargar detalle ingredientes recetas (CSV)",
        data=pd.DataFrame(detail_rows).to_csv(index=False).encode("utf-8"),
        file_name="detalle_recetas.csv",
        mime="text/csv"
    )

# ==== Footer note ====

st.markdown("""
---
**Note**: Data and recipes are stored only for this browser session.
For persistent storage (file or database), enhancements can be added.
""")
