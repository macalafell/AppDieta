import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
from io import BytesIO
import altair as alt

# ------------------------------
# Utility Functions
# ------------------------------
def norm_txt(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def to_float_series(x: pd.Series) -> pd.Series:
    if x.dtype.kind in {"i", "u", "f"}:
        return x.astype(float)
    return pd.to_numeric(
        x.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def kcal_from_macros(carb_g, prot_g, fat_g):
    return carb_g * 4.0 + prot_g * 4.0 + fat_g * 9.0


def mifflin_st_jeor_bmr(sex, weight, height, age):
    male_tokens = {"male", "man", "hombre", "masculino", "varon", "varón"}
    sex_n = norm_txt(sex)
    return 10 * weight + 6.25 * height - 5 * age + (5 if sex_n in male_tokens else -161)

# ------------------------------
# NNLS Solver for grams adjustment
# ------------------------------

def nnls_iterative(A, b, max_iter=50):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)
    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale == 0] = 1.0
    A_s = A / col_scale
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
    x = np.maximum(0.0, x) / col_scale
    return x

# ------------------------------
# Data Loader with caching
# ------------------------------

@st.cache_data(show_spinner=False)
def load_foods(uploaded):
    def find_first(cols_map, keys):
        for k in keys:
            if k in cols_map:
                return cols_map[k]
        return None

    def normalize_df(df):
        EXPECTED = {
            "name": ["producto", "alimento", "nombre"],
            "brand": ["marca"],
            "cat": ["categoria", "categoría"],
            "subcat": ["subcategoria", "subcategoría"],
            "kcal_g": ["energia (kcal/g)", "kcal/g"],
            "kcal_100g": ["energia (kcal/100g)", "kcal/100g"],
            "carb_g": ["carbohidratos (g/g)", "hidratos (g/g)", "carbs (g/g)"],
            "carb_100g": ["carbohidratos (g/100g)", "hidratos (g/100g)", "carbs (g/100g)"],
            "prot_g": ["proteinas (g/g)", "proteínas (g/g)", "protein (g/g)"],
            "prot_100g": ["proteinas (g/100g)", "proteínas (g/100g)", "protein (g/100g)"],
            "fat_g": ["grasas (g/g)", "lipidos (g/g)"],
            "fat_100g": ["grasas (g/100g)", "lipidos (g/100g)"]
        }
        cols_map = {norm_txt(c): c for c in df.columns}
        name_col = find_first(cols_map, EXPECTED["name"])
        brand_col = find_first(cols_map, EXPECTED["brand"])
        cat_col = find_first(cols_map, EXPECTED["cat"])
        subcat_col = find_first(cols_map, EXPECTED["subcat"])
        kcal_g_src = find_first(cols_map, EXPECTED["kcal_g"])
        kcal_100_src = find_first(cols_map, EXPECTED["kcal_100g"])
        carb_g_src = find_first(cols_map, EXPECTED["carb_g"])
        carb_100_src = find_first(cols_map, EXPECTED["carb_100g"])
        prot_g_src = find_first(cols_map, EXPECTED["prot_g"])
        prot_100_src = find_first(cols_map, EXPECTED["prot_100g"])
        fat_g_src = find_first(cols_map, EXPECTED["fat_g"])
        fat_100_src = find_first(cols_map, EXPECTED["fat_100g"])

        kcal_g = to_float_series(df[kcal_g_src]) if kcal_g_src else (
            to_float_series(df[kcal_100_src]) / 100 if kcal_100_src else None
        )
        carb_g = to_float_series(df[carb_g_src]) if carb_g_src else (
            to_float_series(df[carb_100_src]) / 100 if carb_100_src else None
        )
        prot_g = to_float_series(df[prot_g_src]) if prot_g_src else (
            to_float_series(df[prot_100_src]) / 100 if prot_100_src else None
        )
        fat_g = to_float_series(df[fat_g_src]) if fat_g_src else (
            to_float_series(df[fat_100_src]) / 100 if fat_100_src else None
        )

        out = pd.DataFrame({
            "Producto": df[name_col].astype(str),
            "Marca": df[brand_col].astype(str) if brand_col else "",
            "Categoría": df[cat_col].astype(str) if cat_col else "",
            "Subcategoría": df[subcat_col].astype(str) if subcat_col else "",
            "carb_g": carb_g, "prot_g": prot_g, "fat_g": fat_g, "kcal_g": kcal_g
        })
        missing_kcal = out["kcal_g"].isna()
        has_macros = out[["carb_g", "prot_g", "fat_g"]].notna().all(axis=1)
        out.loc[missing_kcal & has_macros, "kcal_g"] = out.loc[
            missing_kcal & has_macros, ["carb_g", "prot_g", "fat_g"]
        ].apply(lambda row: kcal_from_macros(row[0], row[1], row[2]), axis=1)
        return out.dropna(subset=["kcal_g", "carb_g", "prot_g", "fat_g"]).reset_index(drop=True)

    try:
        if uploaded is not None:
            xls = pd.ExcelFile(BytesIO(uploaded.read()))
        elif os.path.exists("alimentos_800_especificos.xlsx"):
            xls = pd.ExcelFile("alimentos_800_especificos.xlsx")
        else:
            return pd.DataFrame()
        sheet = "Todos" if "Todos" in xls.sheet_names else xls.sheet_names[0]
        return normalize_df(pd.read_excel(xls, sheet_name=sheet))
    except Exception as e:
        st.error(f"Could not read food data: {e}")
        return pd.DataFrame()

# ------------------------------
# Sidebar UI
# ------------------------------

def sidebar_ui():
    st.sidebar.header("Profile & parameters")
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=65.0, step=0.5)
    height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=178.0, step=0.5)
    age = st.sidebar.number_input("Age (years)", min_value=14, max_value=100, value=35, step=1)
    st.sidebar.markdown("---")
    cal_mode = st.sidebar.radio("Mode", ["Multiplier", "Manual kcal"], horizontal=True)
    if cal_mode == "Multiplier":
        mult_high = st.sidebar.number_input("High", value=1.60, step=0.01)
        mult_medium = st.sidebar.number_input("Medium", value=1.55, step=0.01)
        mult_low = st.sidebar.number_input("Low", value=1.50, step=0.01)
        extra_high = extra_medium = extra_low = 0.0
    else:
        extra_high = st.sidebar.number_input("Extra kcal - HIGH", value=0, step=10, min_value=-2000, max_value=2000)
        extra_medium = st.sidebar.number_input("Extra kcal - MEDIUM", value=0, step=10, min_value=-2000, max_value=2000)
        extra_low = st.sidebar.number_input("Extra kcal - LOW", value=0, step=10, min_value=-2000, max_value=2000)
        mult_high = mult_medium = mult_low = 1.0
    st.sidebar.markdown("---")
    p_high = st.sidebar.number_input("Protein (g/kg) - HIGH", value=1.4, step=0.1)
    g_high = st.sidebar.number_input("Fat (g/kg) - HIGH", value=0.7, step=0.1)
    p_medium = st.sidebar.number_input("Protein (g/kg) - MEDIUM", value=1.7, step=0.1)
    g_medium = st.sidebar.number_input("Fat (g/kg) - MEDIUM", value=1.1, step=0.1)
    p_low = st.sidebar.number_input("Protein (g/kg) - LOW", value=2.0, step=0.1)
    g_low = st.sidebar.number_input("Fat (g/kg) - LOW", value=1.5, step=0.1)
    st.sidebar.markdown("---")
    adj_pct = st.sidebar.slider("Total calories adjustment (%)", min_value=-25, max_value=25, value=-10, step=1)
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload your foods Excel (optional)", type=["xlsx"])

    return {
        "sex": sex, "weight": weight, "height": height, "age": age,
        "cal_mode": cal_mode,
        "mult_high": mult_high, "mult_medium": mult_medium, "mult_low": mult_low,
        "extra_high": extra_high, "extra_medium": extra_medium, "extra_low": extra_low,
        "p_high": p_high, "p_medium": p_medium, "p_low": p_low,
        "g_high": g_high, "g_medium": g_medium, "g_low": g_low,
        "adj_pct": adj_pct, "uploaded": uploaded,
    }

# ------------------------------
# Recipe builder / Menu creator
# ------------------------------

def menu_recipe_builder(foods, p_target, f_target, c_target, kcal_target, meal, day_type):
    st.markdown("### Recipe builder")
    if foods.empty:
        st.warning("Upload an Excel food file or place one in the app folder.")
        return

    df_view = foods.copy().reset_index(drop=True)
    df_view["Marca"] = df_view["Marca"].astype(str).fillna("")
    df_view["__label__"] = np.where(
        df_view["Marca"].str.strip() != "",
        df_view["Producto"] + " (" + df_view["Marca"] + ")",
        df_view["Producto"],
    )
    options = df_view.index.tolist()
    choices_idx = st.multiselect(
        "Pick up to 10 foods for the recipe",
        options=options,
        format_func=lambda i: df_view.loc[i, "__label__"],
    )
    if len(choices_idx) > 10:
        st.warning("Only the first 10 selected items will be used.")
        choices_idx = choices_idx[:10]

    selected = df_view.loc[choices_idx].drop_duplicates("Producto").reset_index(drop=True)
    if not selected.empty:
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
                pd.Series([locks.get(p, False) for p in base_df["Producto"].tolist()], index=base_df.index),
            )
            base_df.insert(2, "Grams (g)", 0.0)
            st.session_state[editor_key] = base_df
            st.session_state[editor_key + "_products"] = current_products
            st.session_state[lock_key] = locks

        editor_df = st.session_state[editor_key]
        st.write("Enter grams (leave 0 to auto-adjust):")
        editor_df = st.data_editor(
            editor_df,
            key=editor_key + "_table",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Producto": st.column_config.TextColumn("Product", disabled=True),
                "Locked": st.column_config.CheckboxColumn(
                    "Lock 🔒",
                    help="Ingredient not changed by auto-adjust.",
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
        locks = st.session_state.get(lock_key, {})
        for _, r in editor_df.iterrows():
            p = r["Producto"]
            g = float(r["Grams (g)"]) if pd.notna(r["Grams (g)"]) else 0.0
            checked = bool(r.get("Locked", False))
            locks[p] = False if g == 0 else checked
        st.session_state[lock_key] = locks

        grams = editor_df["Grams (g)"].to_numpy()
        totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
        kcal_tot = float(totals.get("kcal_g", 0.0))
        carb_tot = float(totals.get("carb_g", 0.0))
        prot_tot = float(totals.get("prot_g", 0.0))
        fat_tot = float(totals.get("fat_g", 0.0))

        colA, colB, colC, colD = st.columns(4)
        colA.metric("kcal", f"{kcal_tot:.0f}", delta=f"{kcal_tot - kcal_target:+.0f}")
        colB.metric("Carbs (g)", f"{carb_tot:.0f}", delta=f"{carb_tot - c_target:+.0f}")
        colC.metric("Protein (g)", f"{prot_tot:.0f}", delta=f"{prot_tot - p_target:+.0f}")
        colD.metric("Fat (g)", f"{fat_tot:.0f}", delta=f"{fat_tot - f_target:+.0f}")

        st.markdown("**Adjust grams**")
        btn_col1, btn_col2 = st.columns([1, 2])

        with btn_col1:
            if st.button("Adjust ALL (match targets)"):
                A_full = editor_df[["carb_g", "prot_g", "fat_g"]].to_numpy().T
                b = np.array([c_target, p_target, f_target], dtype=float)
                products = editor_df["Producto"].tolist()
                grams_now = editor_df["Grams (g)"].to_numpy().astype(float)
                locks = st.session_state.get(lock_key, {p: False for p in products})
                locked_idx = [i for i, p in enumerate(products) if locks.get(p, False)]
                unlocked_idx = [i for i, p in enumerate(products) if not locks.get(p, False)]
                b_res = b
                if locked_idx:
                    A_lock = A_full[:, locked_idx]
                    g_lock = grams_now[locked_idx]
                    b_res = b - A_lock @ g_lock
                if unlocked_idx:
                    A_un = A_full[:, unlocked_idx]
                    x_un = nnls_iterative(A_un, b_res, max_iter=50)
                    new_grams = grams_now.copy()
                    new_grams[unlocked_idx] = x_un
                    editor_df.loc[:, "Grams (g)"] = new_grams
                    st.session_state[editor_key] = editor_df
                    st.success("Adjusted grams for all unlocked ingredients.")
                    st.experimental_rerun()

        with btn_col2:
            ing_choice = st.selectbox(
                "Ingredient to adjust (single)", editor_df["Producto"].tolist(), key=f"single_sel_{meal}"
            )
            if st.button("Adjust ONLY this ingredient"):
                deficits = np.array([c_target - carb_tot, p_target - prot_tot, f_target - fat_tot], dtype=float)
                v = editor_df.loc[editor_df["Producto"] == ing_choice, ["carb_g", "prot_g", "fat_g"]].to_numpy().ravel()
                denom = float(np.dot(v, v))
                if denom > 0 and np.isfinite(denom):
                    g_delta = float(np.dot(v, deficits)) / denom
                    current_g = float(editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"].iloc[0])
                    new_val = max(0.0, current_g + g_delta)
                    editor_df.loc[editor_df["Producto"] == ing_choice, "Grams (g)"] = new_val
                    st.session_state[editor_key] = editor_df
                    st.success(
                        f"Grams {'increased' if g_delta >= 0 else 'reduced'} for '{ing_choice}' by {abs(g_delta):.0f} g (new total: {new_val:.0f} g)."
                    )

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
        if st.button("Save recipe to session"):
            grams = editor_df["Grams (g)"]
            totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
            r = {
                "nombre": recipe_name or f"Recipe {len(st.session_state['recipes']) + 1}",
                "tipo_dia": day_type,
                "comida": meal,
                "objetivo": {
                    "kcal": float(kcal_target),
                    "carb": float(c_target),
                    "prot": float(p_target),
                    "fat": float(f_target),
                },
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
            st.session_state["recipes"].append(r)
            st.success("Recipe saved in the session.")

# ------------------------------
# Main App
# ------------------------------

st.set_page_config(page_title="DIET APP · Meal Planner", layout="wide")
st.title("APP Recipe Builder")

# Sidebar profile and parameters
params = sidebar_ui()

# Load foods data
foods = load_foods(params["uploaded"])

# Calculate BMR and TDEE parameters
bmr = mifflin_st_jeor_bmr(params["sex"], params["weight"], params["height"], params["age"])
day_type = st.selectbox("Day type", ["High", "Medium", "Low"])
mults = {"High": params["mult_high"], "Medium": params["mult_medium"], "Low": params["mult_low"]}
extras = {"High": params["extra_high"], "Medium": params["extra_medium"], "Low": params["extra_low"]}

if params["cal_mode"] == "Multiplier":
    base_tdee = bmr * mults[day_type]
else:
    base_tdee = bmr + extras[day_type]

tdee = base_tdee * (1 + params["adj_pct"] / 100.0)

# Daily macros in grams
protein_g = {"High": params["p_high"], "Medium": params["p_medium"], "Low": params["p_low"]}[day_type] * params["weight"]
fat_g = {"High": params["g_high"], "Medium": params["g_medium"], "Low": params["g_low"]}[day_type] * params["weight"]
kcal_p_f = protein_g * 4 + fat_g * 9
carbs_g = max(0.0, (tdee - kcal_p_f) / 4.0)

# Show metrics and macros pie chart
col1, col2, col3 = st.columns([1, 1, 1.2])
with col1:
    st.metric("BMR (kcal/day)", f"{bmr:.0f}")
    st.metric("TDEE (kcal/day)", f"{tdee:.0f}")
with col2:
    st.metric("Protein (g/day)", f"{protein_g:.0f}")
    st.metric("Fat (g/day)", f"{fat_g:.0f}")
    st.metric("Carbohydrates (g/day)", f"{carbs_g:.0f}")
with col3:
    macros_df = pd.DataFrame({
        "Macro": ["Carbohydrates", "Protein", "Fat"],
        "Grams": [carbs_g, protein_g, fat_g],
        "kcal": [carbs_g * 4, protein_g * 4, fat_g * 9],
    })
    macros_df["% kcal"] = (macros_df["kcal"] / macros_df["kcal"].sum() * 100).round(1)
    colors = {"Carbohydrates": "#EE9B00", "Protein": "#CA6702", "Fat": "#BB3E03"}
    try:
        chart = (alt.Chart(macros_df)
                 .mark_arc()
                 .encode(
                     theta=alt.Theta("kcal:Q"),
                     color=alt.Color("Macro:N", scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values()))),
                     tooltip=["Macro", "Grams", "kcal", "% kcal"],
                 )
                 .properties(width=360, height=360))
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(macros_df.set_index("Macro")["kcal"])

# Meal selection and defaults for split
meal_split = {
    "Breakfast": {"prot": 0.10, "fat": 0.10, "carb": 0.27},
    "Lunch": {"prot": 0.39, "fat": 0.40, "carb": 0.26},
    "Snack": {"prot": 0.08, "fat": 0.06, "carb": 0.17},
    "Dinner": {"prot": 0.43, "fat": 0.44, "carb": 0.30},
}

st.markdown("### Meal split")
with st.expander("Edit split (portion of the day)"):
    for m in meal_split:
        meal_split[m]["prot"] = st.number_input(f"Protein ({m})", value=meal_split[m]["prot"], step=0.01, format="%.2f", key=f"prot_{m}")
        meal_split[m]["fat"] = st.number_input(f"Fat ({m})", value=meal_split[m]["fat"], step=0.01, format="%.2f", key=f"fat_{m}")
        meal_split[m]["carb"] = st.number_input(f"Carbs ({m})", value=meal_split[m]["carb"], step=0.01, format="%.2f", key=f"carb_{m}")
totals = {k: sum(meal_split[m][k] for m in meal_split) for k in ("prot", "fat", "carb")}
warn_msgs = [f"{k} sum = {v:.2f}" for k, v in totals.items() if not (0.95 <= v <= 1.05)]
if warn_msgs:
    st.warning("; ".join(warn_msgs) + ". Ideally each macro sum ≈ 1.00 across all meals.")

# Calculate macros targets per meal
meal = st.selectbox("Select Meal", list(meal_split.keys()))
p_target = protein_g * meal_split[meal]["prot"]
f_target = fat_g * meal_split[meal]["fat"]
c_target = carbs_g * meal_split[meal]["carb"]
kcal_target = p_target * 4 + f_target * 9 + c_target * 4
st.info(f"Target for {meal} → {kcal_target:.0f} kcal | Protein: {p_target:.0f} g | Fat: {f_target:.0f} g | Carbs: {c_target:.0f} g")

# Call the recipe/menu builder function
menu_recipe_builder(foods, p_target, f_target, c_target, kcal_target, meal, day_type)

# Saved recipes display
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
                    st.write(
                        f"{r['objetivo']['kcal']:.0f} kcal | C:{r['objetivo']['carb']:.0f} g · P:{r['objetivo']['prot']:.0f} g · F:{r['objetivo']['fat']:.0f} g"
                    )
                with col2:
                    st.write("**Result**")
                    st.write(
                        f"{r['resultado']['kcal']:.0f} kcal | C:{r['resultado']['carb']:.0f} g · P:{r['resultado']['prot']:.0f} g · F:{r['resultado']['fat']:.0f} g"
                    )
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

# Export all saved recipes button
if st.session_state.get("recipes"):
    st.markdown("#### Export ALL recipes (session)")
    buf_all = BytesIO()
    with pd.ExcelWriter(buf_all, engine="openpyxl") as writer:
        summary_rows = []
        for r in st.session_state["recipes"]:
            summary_rows.append({
                "Name": r["nombre"], "Day type": r["tipo_dia"], "Meal": r["comida"],
                "kcal_target": r["objetivo"]["kcal"], "carb_target": r["objetivo"]["carb"],
                "prot_target": r["objetivo"]["prot"], "fat_target": r["objetivo"]["fat"],
                "kcal_result": r["resultado"]["kcal"], "carb_result": r["resultado"]["carb"],
                "prot_result": r["resultado"]["prot"], "fat_result": r["resultado"]["fat"],
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

st.markdown("""
---
**Note**: Data and recipes are stored only for this browser session.  
Use the Excel download buttons above to save your work locally.
""")
