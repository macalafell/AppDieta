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
    return pd.to_numeric(x.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False), errors="coerce")

def kcal_from_macros(carb_g, prot_g, fat_g):
    return carb_g * 4.0 + prot_g * 4.0 + fat_g * 9.0

def mifflin_st_jeor_bmr(sex, weight, height, age):
    male_tokens = {"male", "man", "hombre", "masculino", "varon", "varón"}
    sex_n = norm_txt(sex)
    return 10 * weight + 6.25 * height - 5 * age + (5 if sex_n in male_tokens else -161)

# ------------------------------
# Data Loading with Caching
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
        ].apply(lambda row: kcal_from_macros(row, row[2], row[21]), axis=1)
        return out.dropna(subset=["kcal_g", "carb_g", "prot_g", "fat_g"]).reset_index(drop=True)

    if uploaded is not None:
        xls = pd.ExcelFile(BytesIO(uploaded.read()))
    elif os.path.exists("alimentos_800_especificos.xlsx"):
        xls = pd.ExcelFile("alimentos_800_especificos.xlsx")
    else:
        return pd.DataFrame()

    sheet = "Todos" if "Todos" in xls.sheet_names else xls.sheet_names
    return normalize_df(pd.read_excel(xls, sheet_name=sheet))

# ------------------------------
# Sidebar UI - Profile & Params
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
    return locals()

# ------------------------------
# Main App
# ------------------------------
st.set_page_config(page_title="DIET APP · Meal Planner", layout="wide")
st.title("APP Recipe Builder")
params = sidebar_ui()
foods = load_foods(params["uploaded"])
bmr = mifflin_st_jeor_bmr(params["sex"], params["weight"], params["height"], params["age"])
day_type = st.selectbox("Day type", ["High", "Medium", "Low"])
mults = {"High": params["mult_high"], "Medium": params["mult_medium"], "Low": params["mult_low"]}
extras = {"High": params["extra_high"], "Medium": params["extra_medium"], "Low": params["extra_low"]}
if params["cal_mode"] == "Multiplier":
    base_tdee = bmr * mults[day_type]
else:
    base_tdee = bmr + extras[day_type]
tdee = base_tdee * (1 + params["adj_pct"] / 100.0)
macro_vals = {
    "Protein": {"High": params["p_high"], "Medium": params["p_medium"], "Low": params["p_low"]},
    "Fat": {"High": params["g_high"], "Medium": params["g_medium"], "Low": params["g_low"]}
}
p_day = macro_vals["Protein"][day_type] * params["weight"]
f_day = macro_vals["Fat"][day_type] * params["weight"]
kcal_from_p_f = p_day * 4 + f_day * 9
c_day = max(0.0, (tdee - kcal_from_p_f) / 4.0)

# Metrics and Chart
col1, col2, col3 = st.columns([1, 1, 1.2])
with col1:
    st.metric("BMR (kcal/day)", f"{bmr:.0f}")
    st.metric("TDEE (kcal/day)", f"{tdee:.0f}")
with col2:
    st.metric("Protein (g/day)", f"{p_day:.0f}")
    st.metric("Fat (g/day)", f"{f_day:.0f}")
    st.metric("Carbohydrates (g/day)", f"{c_day:.0f}")
with col3:
    macros_daily_df = pd.DataFrame(
        {"Macro": ["Carbohydrates", "Protein", "Fat"],
         "Grams": [c_day, p_day, f_day],
         "kcal": [c_day * 4, p_day * 4, f_day * 9]}
    )
    macros_daily_df["% kcal"] = (macros_daily_df["kcal"] / macros_daily_df["kcal"].sum() * 100).round(1)
    macro_colors = {"Carbohydrates": "#EE9B00", "Protein": "#CA6702", "Fat": "#BB3E03"}
    try:
        pie = (alt.Chart(macros_daily_df)
                .mark_arc()
                .encode(
                    theta=alt.Theta("kcal:Q"),
                    color=alt.Color("Macro:N", scale=alt.Scale(domain=list(macro_colors.keys()), range=list(macro_colors.values()))),
                    tooltip=["Macro", "Grams", "kcal", "% kcal"]
                ).properties(width=360, height=360))
        st.altair_chart(pie, use_container_width=True)
    except Exception:
        st.bar_chart(macros_daily_df.set_index("Macro")["kcal"])

# Rest of your recipe builder logic (modularize meal split, editing, export logic to helper functions for further clarity)
# End Note
st.markdown("""
---
**Note**: Data and recipes are stored only for this browser session.  
Use the Excel download buttons above to save your work locally.
""")
