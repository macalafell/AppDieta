import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
from io import BytesIO
from typing import Dict, List, Optional
import altair as alt


def _safe_rerun():
    """Compatibilidad Streamlit: usa st.rerun() si existe; si no, st.experimental_rerun()."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# =============================
# Configuración de la página
# =============================
st.set_page_config(page_title="APP DIETA · Planificador de platos", layout="wide")

# === Estilos globales (títulos #BB4430) ===
st.markdown("""
<style>
  h1, h2, h3 { color: #BB4430 !important; }
  /* Evitar solapes en encabezado del expander */
  div.streamlit-expanderHeader { white-space: normal !important; overflow: hidden; }
  div.streamlit-expanderHeader p { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("APP Creador Recetas")


# =============================
# Utilidades generales
# =============================

def _norm_txt(s: str) -> str:
    """Normaliza texto para matching de columnas: minúsculas, sin acentos, espacios colapsados."""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def _to_float_series(x: pd.Series) -> pd.Series:
    """Convierte una serie a float de forma robusta (soporta '3,4', espacios, etc.)."""
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
# Carga y normalización de Excel
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
        raise ValueError("No se encontró columna de nombre del producto (p. ej., 'Producto'/'Alimento').")
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
        st.error(f"No se pudo leer el Excel: {e}")
        return pd.DataFrame()


# =============================
# Solver NNLS (sin SciPy) para ajustar gramos de TODOS los alimentos
# =============================

def nnls_iterative(A: np.ndarray, b: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Resuelve min ||A x - b|| con x>=0 (active set simple) y escala de columnas.
    - A: matriz (m x n) con densidades por gramo (p.ej., filas=carb/prot/fat; columnas=alimentos)
    - b: vector objetivo (m,)
    Devuelve x (n,) en gramos por alimento.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)

    # Escalado de columnas para mejorar acondicionamiento
    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale == 0] = 1.0
    A_s = A / col_scale

    # Solución LS y poda de negativos iterativamente
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

    # Des-escalado
    x = np.maximum(0.0, x) / col_scale
    return x

# =============================
# Sidebar: Perfil y parámetros
# =============================

st.sidebar.header("Perfil y parámetros")
sex = st.sidebar.selectbox("Sexo", ["Hombre", "Mujer"])
weight = st.sidebar.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.5)
height = st.sidebar.number_input("Altura (cm)", min_value=120.0, max_value=230.0, value=178.0, step=0.5)
age = st.sidebar.number_input("Edad (años)", min_value=14, max_value=100, value=35, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Objetivo calórico por tipo de día")
cal_mode = st.sidebar.radio("Modo", ["Multiplicador", "Kcal manual"], horizontal=True)

if cal_mode == "Multiplicador":
    mult_alto = st.sidebar.number_input("Alto", value=1.60, step=0.01, format="%.2f")
    mult_medio = st.sidebar.number_input("Medio", value=1.55, step=0.01, format="%.2f")
    mult_bajo = st.sidebar.number_input("Bajo", value=1.50, step=0.01, format="%.2f")
    # Valores por defecto para extras (no usados en este modo)
    extra_alto = extra_medio = extra_bajo = 0.0
else:
    st.sidebar.caption("Añade o resta kcal al BMR por tipo de día")
    extra_alto = st.sidebar.number_input("Kcal extra - ALTO", value=0, step=10, min_value=-2000, max_value=2000)
    extra_medio = st.sidebar.number_input("Kcal extra - MEDIO", value=0, step=10, min_value=-2000, max_value=2000)
    extra_bajo = st.sidebar.number_input("Kcal extra - BAJO", value=0, step=10, min_value=-2000, max_value=2000)
    # Valores por defecto para multiplicadores (no usados en este modo)
    mult_alto = mult_medio = mult_bajo = 1.0

st.sidebar.markdown("---")
st.sidebar.subheader("Macros diarios por tipo de día (g/kg de peso)")
st.sidebar.caption("Día ALTO")
p_alto = st.sidebar.number_input("Proteína (g/kg) - ALTO", value=1.4, step=0.1)
g_alto = st.sidebar.number_input("Grasa (g/kg) - ALTO", value=0.7, step=0.1)
st.sidebar.caption("Día MEDIO")
p_medio = st.sidebar.number_input("Proteína (g/kg) - MEDIO", value=1.7, step=0.1)
g_medio = st.sidebar.number_input("Grasa (g/kg) - MEDIO", value=1.1, step=0.1)
st.sidebar.caption("Día BAJO")
p_bajo = st.sidebar.number_input("Proteína (g/kg) - BAJO", value=2.0, step=0.1)
g_bajo = st.sidebar.number_input("Grasa (g/kg) - BAJO", value=1.5, step=0.1)

st.sidebar.markdown("---")
adj_pct = st.sidebar.slider("Ajuste de calorías totales (%)", min_value=-25, max_value=25, value=0, step=1)

# Carbohidratos autocalculados en g/kg
st.sidebar.markdown("---")
st.sidebar.subheader("Carbohidratos (g/kg) calculados")

def _tdee_by_method(day_label: str, adj_pct_value: float) -> float:
    bmr_local = mifflin_st_jeor_bmr(sex, weight, height, age)
    if cal_mode == "Multiplicador":
        mult_map = {"Alto": mult_alto, "Medio": mult_medio, "Bajo": mult_bajo}
        base = bmr_local * mult_map[day_label]
    else:
        extra_map = {"Alto": float(extra_alto), "Medio": float(extra_medio), "Bajo": float(extra_bajo)}
        base = bmr_local + extra_map[day_label]
    return base * (1 + adj_pct_value / 100.0)


def carbs_g_per_kg_for_day(day_label: str, p_gkg: float, f_gkg: float, adj_pct_value: float) -> float:
    tdee_x = _tdee_by_method(day_label, adj_pct_value)
    p_day_x = p_gkg * weight
    f_day_x = f_gkg * weight
    c_day_x = max(0.0, (tdee_x - (p_day_x * 4 + f_day_x * 9)) / 4.0)  # g/día
    return round(float(c_day_x / weight), 2)  # g/kg

carbs_alto_gkg = carbs_g_per_kg_for_day("Alto", p_alto, g_alto, adj_pct)
carbs_medio_gkg = carbs_g_per_kg_for_day("Medio", p_medio, g_medio, adj_pct)
carbs_bajo_gkg = carbs_g_per_kg_for_day("Bajo", p_bajo, g_bajo, adj_pct)

st.sidebar.caption("Día ALTO")
st.sidebar.number_input("Carbohidratos (g/kg) - ALTO", value=carbs_alto_gkg, step=0.01, format="%.2f", disabled=True, key="c_alto_gkg_ro")
st.sidebar.caption("Día MEDIO")
st.sidebar.number_input("Carbohidratos (g/kg) - MEDIO", value=carbs_medio_gkg, step=0.01, format="%.2f", disabled=True, key="c_medio_gkg_ro")
st.sidebar.caption("Día BAJO")
st.sidebar.number_input("Carbohidratos (g/kg) - BAJO", value=carbs_bajo_gkg, step=0.01, format="%.2f", disabled=True, key="c_bajo_gkg_ro")

# Tabla integrada de g/kg por tipo de día
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Sube tu Excel de alimentos (opcional)", type=["xlsx"])

# =============================
# Cargar alimentos
# =============================
foods = load_foods(uploaded)

# =============================
# Cálculos diarios (g/día a partir de g/kg y TDEE)
# =============================

bmr = mifflin_st_jeor_bmr(sex, weight, height, age)

# Selección de tipo de día y TDEE según modo
tipo_dia = st.selectbox("Tipo de día", ["Alto", "Medio", "Bajo"])
if cal_mode == "Multiplicador":
    mult_map = {"Alto": mult_alto, "Medio": mult_medio, "Bajo": mult_bajo}
    base_tdee = bmr * mult_map[tipo_dia]
else:
    extra_map = {"Alto": float(extra_alto), "Medio": float(extra_medio), "Bajo": float(extra_bajo)}
    base_tdee = bmr + extra_map[tipo_dia]

tdee = base_tdee * (1 + adj_pct / 100.0)

# Macros diarios objetivo (g/día)
p_day = {"Alto": p_alto, "Medio": p_medio, "Bajo": p_bajo}[tipo_dia] * weight
f_day = {"Alto": g_alto, "Medio": g_medio, "Bajo": g_bajo}[tipo_dia] * weight
kcal_from_p_f = p_day * 4 + f_day * 9
c_day = max(0.0, (tdee - kcal_from_p_f) / 4.0)

left, right = st.columns([1, 1])
with left:
    st.metric("BMR (kcal/día)", f"{bmr:.0f}")
    st.metric("TDEE (kcal/día)", f"{tdee:.0f}")
with right:
    st.metric("Proteína (g/día)", f"{p_day:.0f}")
    st.metric("Grasa (g/día)", f"{f_day:.0f}")
    st.metric("Carbohidratos (g/día)", f"{c_day:.0f}")

# =============================
# Gráfico: Reparto de macronutrientes diarios
# =============================

st.markdown("### Reparto de macronutrientes diarios")
macros_daily_df = pd.DataFrame(
    {
        "Macro": ["Carbohidratos", "Proteína", "Grasa"],
        "Gramos": [c_day, p_day, f_day],
        "kcal": [c_day * 4, p_day * 4, f_day * 9],
    }
)
macros_daily_df["% kcal"] = (macros_daily_df["kcal"] / macros_daily_df["kcal"].sum() * 100).round(1)

try:
    pie = (
        alt.Chart(macros_daily_df)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="kcal", type="quantitative"),
            color=alt.Color(
                field="Macro",
                type="nominal",
                # Colores pedidos: 7EBDC2, F3DFA2, EFE6DD
                scale=alt.Scale(
                    domain=["Carbohidratos", "Proteína", "Grasa"],
                    range=["#7EBDC2", "#F3DFA2", "#EFE6DD"],
                ),
            ),
            tooltip=["Macro", "Gramos", "kcal", "% kcal"],
        )
        .properties(width=360, height=360)
    )
    st.altair_chart(pie, use_container_width=False)
except Exception:
    st.bar_chart(macros_daily_df.set_index("Macro")["kcal"])

# =============================
# Reparto por comida (editable)
# =============================

st.markdown("### Reparto por comida")
meal_defaults = {
    "Desayuno": {"prot": 0.10, "fat": 0.10, "carb": 0.27},
    "Almuerzo": { "prot": 0.39, "fat": 0.40, "carb": 0.26},
    "Merienda": {"prot": 0.08, "fat": 0.06, "carb": 0.17},
    "Cena": {"prot": 0.43, "fat": 0.44, "carb": 0.30},
}

with st.expander("Editar reparto (proporción del día)"):
    for key in meal_defaults:
        st.write(f"**{key}**")
        meal_defaults[key]["prot"] = st.number_input(
            f"Proteína ({key})", value=float(meal_defaults[key]["prot"]), step=0.01, format="%.2f", key=f"p_{key}"
        )
        meal_defaults[key]["fat"] = st.number_input(
            f"Grasa ({key})", value=float(meal_defaults[key]["fat"]), step=0.01, format="%.2f", key=f"g_{key}"
        )
        meal_defaults[key]["carb"] = st.number_input(
            f"Hidratos ({key})", value=float(meal_defaults[key]["carb"]), step=0.01, format="%.2f", key=f"c_{key}"
        )

    totals = {k: sum(meal_defaults[m][k] for m in meal_defaults) for k in ("prot", "fat", "carb")}
    warn_msgs = []
    for k, v in totals.items():
        if not (0.95 <= v <= 1.05):
            warn_msgs.append(f"suma {k} = {v:.2f}")
    if warn_msgs:
        st.warning("; ".join(warn_msgs) + ". Idealmente cada macro debe sumar ≈ 1.00 entre todas las comidas.")

# =============================
# Resumen de macros por comida + exportar Excel (con totales)
# =============================

st.markdown("### Resumen de macros por comida")

def meal_targets(meal_name: str, perc: Dict[str, float]) -> Dict[str, float]:
    p_t = p_day * perc["prot"]
    f_t = f_day * perc["fat"]
    c_t = c_day * perc["carb"]
    kcal_t = c_t * 4 + p_t * 4 + f_t * 9
    return {
        "Comida": meal_name,
        "kcal": round(kcal_t, 0),
        "Carbohidratos (g)": round(c_t, 1),
        "Proteína (g)": round(p_t, 1),
        "Grasa (g)": round(f_t, 1),
    }

meals_summary = pd.DataFrame([
    meal_targets("Desayuno", meal_defaults["Desayuno"]),
    meal_targets("Almuerzo", meal_defaults["Almuerzo"]),
    meal_targets("Merienda", meal_defaults["Merienda"]),
    meal_targets("Cena", meal_defaults["Cena"]),
])

# Añadir fila TOTAL
total_row = {
    "Comida": "TOTAL",
    "kcal": round(meals_summary["kcal"].sum(), 0),
    "Carbohidratos (g)": round(meals_summary["Carbohidratos (g)"].sum(), 1),
    "Proteína (g)": round(meals_summary["Proteína (g)"].sum(), 1),
    "Grasa (g)": round(meals_summary["Grasa (g)"].sum(), 1),
}
meals_summary_tot = pd.concat([meals_summary, pd.DataFrame([total_row])], ignore_index=True)

# --- Render HTML minimalista con negritas en columna Comida y fila TOTAL ---
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
        font-weight: 700; /* Columna Comida en negrita */
      }
      table.meals-summary tr:last-child td {
        font-weight: 700; /* Fila TOTAL en negrita */
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
    for i, row in df.iterrows():
        html.append("<tr>")
        for j, c in enumerate(cols):
            val = row[c]
            if isinstance(val, float):
                # formateo simple
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
    meals_summary_tot.to_excel(writer, index=False, sheet_name="Macros por comida")
buf_meals.seek(0)
st.download_button(
    "Descargar resumen por comida (Excel)",
    data=buf_meals,
    file_name="resumen_macros_por_comida.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =============================
# Objetivos por comida seleccionada
# =============================

st.markdown("### Comida")
meal = st.selectbox("", ["Desayuno", "Almuerzo", "Merienda", "Cena"], label_visibility="collapsed")
perc = meal_defaults[meal]
p_target = p_day * perc["prot"]
f_target = f_day * perc["fat"]
c_target = c_day * perc["carb"]
kcal_target = c_target * 4 + p_target * 4 + f_target * 9
st.info(f"Objetivo para {meal} → {kcal_target:.0f} kcal | Prot: {p_target:.0f} g | Grasa: {f_target:.0f} g | Hidratos: {c_target:.0f} g")

# =============================
# Creador de receta
# =============================

st.markdown("### Creador de receta")
if foods.empty:
    st.warning("Primero sube o coloca en la carpeta un Excel de alimentos (alimentos_800_especificos.xlsx).")
else:
    # (Se ocultó el explorador de filtros + tabla)
    df_view = foods.copy()

    # Multiselección (máximo 10)
    choices = st.multiselect(
        "Elige hasta 10 alimentos para la receta",
        df_view["Producto"].tolist(),
    )
    if len(choices) > 10:
        st.warning("Has seleccionado más de 10 elementos; se usarán los 10 primeros.")
        choices = choices[:10]

    selected = (
        df_view[df_view["Producto"].isin(choices)]
        .drop_duplicates("Producto")
        .reset_index(drop=True)
    )

    if not selected.empty:
        # Editor de gramos con persistencia en session_state
        editor_key = f"editor_{meal}"
        lock_key = f"{editor_key}_locked"
        current_products = selected["Producto"].tolist()
        prev_products = st.session_state.get(editor_key + "_products")

        if prev_products != current_products:
            # inicializar o re-sincronizar (con columna de bloqueo)
            base_df = selected[["Producto", "carb_g", "prot_g", "fat_g", "kcal_g"]].copy()
            old_locks = st.session_state.get(lock_key, {})
            locks = {p: bool(old_locks.get(p, False)) for p in base_df["Producto"].tolist()}
            base_df.insert(1, "Bloqueado", pd.Series([locks[p] for p in base_df["Producto"]]))
            base_df.insert(2, "Gramos (g)", 0.0)
            st.session_state[editor_key] = base_df
            st.session_state[editor_key + "_products"] = current_products
            st.session_state[lock_key] = locks

        editor_df = st.session_state[editor_key]
        # Asegurar columna 'Bloqueado' existe (migración de sesiones antiguas)
        if "Bloqueado" not in editor_df.columns:
            locks = st.session_state.get(lock_key, {p: False for p in editor_df["Producto"].tolist()})
            editor_df.insert(1, "Bloqueado", editor_df["Producto"].map(lambda p: bool(locks.get(p, False))))
            st.session_state[editor_key] = editor_df

        st.write("Introduce gramos (puedes dejar a 0 y usar el ajuste automático):")
        editor_df = st.data_editor(
            editor_df,
            key=editor_key + "_table",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Producto": st.column_config.TextColumn(disabled=True),
                "Bloqueado": st.column_config.CheckboxColumn(
                    "Bloquear 🔒",
                    help="Si está marcado, este ingrediente no se ajustará en el ajuste automático.",
                    default=False,
                ),
                "carb_g": st.column_config.NumberColumn("carb/g", help="Carbohidratos por gramo", disabled=True),
                "prot_g": st.column_config.NumberColumn("prot/g", help="Proteínas por gramo", disabled=True),
                "fat_g": st.column_config.NumberColumn("fat/g", help="Grasas por gramo", disabled=True),
                "kcal_g": st.column_config.NumberColumn("kcal/g", help="kcal por gramo", disabled=True),
                "Gramos (g)": st.column_config.NumberColumn(step=5.0, min_value=0.0),
            },
        )
        st.session_state[editor_key] = editor_df
        # Sincronizar locks desde la tabla (auto-desbloquear si gramos == 0)
        locks = st.session_state.get(lock_key, {})
        for _, r in editor_df.iterrows():
            p = r["Producto"]
            g = float(r["Gramos (g)"]) if pd.notna(r["Gramos (g)"]) else 0.0
            checked = bool(r.get("Bloqueado", False))
            locks[p] = False if g == 0 else checked
        st.session_state[lock_key] = locks

        # Cálculo de totales actuales
        grams = editor_df["Gramos (g)"].to_numpy()
        totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
        kcal_tot = float(totals["kcal_g"]) if not np.isnan(totals["kcal_g"]) else 0.0
        carb_tot = float(totals["carb_g"]) if not np.isnan(totals["carb_g"]) else 0.0
        prot_tot = float(totals["prot_g"]) if not np.isnan(totals["prot_g"]) else 0.0
        fat_tot = float(totals["fat_g"]) if not np.isnan(totals["fat_g"]) else 0.0

        colA, colB, colC, colD = st.columns(4)
        colA.metric("kcal", f"{kcal_tot:.0f}", delta=f"{kcal_tot - kcal_target:+.0f}")
        colB.metric("Carbohidratos (g)", f"{carb_tot:.0f}", delta=f"{carb_tot - c_target:+.0f}")
        colC.metric("Proteínas (g)", f"{prot_tot:.0f}", delta=f"{prot_tot - p_target:+.0f}")
        colD.metric("Grasas (g)", f"{fat_tot:.0f}", delta=f"{fat_tot - f_target:+.0f}")

        # Ajuste de gramos
        st.markdown("**Ajuste de gramos**")
        btn_col1, btn_col2 = st.columns([1, 2])

        # 1) Ajustar TODOS los ingredientes (cuadrar macros totales)
        with btn_col1:
            if st.button("Ajustar TODOS (cuadrar macros)"):
                A_full = editor_df[["carb_g", "prot_g", "fat_g"]].to_numpy().T  # 3 x N
                b = np.array([c_target, p_target, f_target], dtype=float)
                products = editor_df["Producto"].tolist()
                grams_now = editor_df["Gramos (g)"].to_numpy().astype(float)
                locks = st.session_state.get(lock_key, {p: False for p in products})
                locked_idx = [i for i, p in enumerate(products) if locks.get(p, False)]
                unlocked_idx = [i for i, p in enumerate(products) if not locks.get(p, False)]

                if len(unlocked_idx) == 0:
                    st.info("Todos los ingredientes están bloqueados (editados manualmente). Pon alguno a 0 para desbloquearlo y poder ajustarlo.")
                else:
                    # Restar contribución de los bloqueados al objetivo
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
                    editor_df.loc[:, "Gramos (g)"] = new_grams
                    st.session_state[editor_key] = editor_df
                    st.session_state[editor_key + "_programmatic"] = True
                    st.success("Gramos ajustados para los ingredientes desbloqueados.")
                    _safe_rerun()

        # 2) Ajustar SOLO un ingrediente seleccionado por el usuario
        with btn_col2:
            ing_choice = st.selectbox(
                "Ingrediente a ajustar (solo este)", editor_df["Producto"].tolist(), key=f"single_sel_{meal}"
            )
            if st.button("Ajustar SOLO este ingrediente"):
                # déficits actuales respecto al objetivo
                deficits = np.array([
                    c_target - carb_tot,
                    p_target - prot_tot,
                    f_target - fat_tot,
                ], dtype=float)
                v = (
                    editor_df.loc[editor_df["Producto"] == ing_choice, ["carb_g", "prot_g", "fat_g"]]
                    .to_numpy()
                    .ravel()
                )
                denom = float(np.dot(v, v))
                if denom <= 0 or not np.isfinite(denom):
                    st.warning("No se puede ajustar con este ingrediente (densidades no válidas).")
                else:
                    # mejor ajuste LS en una dimensión (puede ser + o -). No permitimos gramos negativos.
                    g_delta = float(np.dot(v, deficits)) / denom
                    current_g = float(
                        editor_df.loc[editor_df["Producto"] == ing_choice, "Gramos (g)"].iloc[0]
                    )
                    new_val = max(0.0, current_g + g_delta)
                    editor_df.loc[editor_df["Producto"] == ing_choice, "Gramos (g)"] = new_val
                    st.session_state[editor_key] = editor_df
                    st.session_state[editor_key + "_programmatic"] = True
                    msg = "aumentados" if g_delta >= 0 else "reducidos"
                    st.success(
                        f"Gramos {msg} en '{ing_choice}' en {abs(g_delta):.0f} g (nuevo total: {new_val:.0f} g)."
                    )

        # Detalle actual de la receta
        df_curr = editor_df[["Producto", "Gramos (g)"]].copy()
        df_curr["Carbohidratos (g)"] = (editor_df["carb_g"] * editor_df["Gramos (g)"]).round(1)
        df_curr["Proteína (g)"] = (editor_df["prot_g"] * editor_df["Gramos (g)"]).round(1)
        df_curr["Grasa (g)"] = (editor_df["fat_g"] * editor_df["Gramos (g)"]).round(1)
        df_curr["kcal"] = (editor_df["kcal_g"] * editor_df["Gramos (g)"]).round(0)

        if not df_curr.empty:
            st.write("**Detalle actual de la receta (antes de guardar)**")
            st.dataframe(df_curr, hide_index=True, use_container_width=True)

        st.markdown("---")
        recipe_name = st.text_input("Nombre de la receta")
        if "recipes" not in st.session_state:
            st.session_state["recipes"] = []

        if st.button("Guardar receta"):
            grams = editor_df["Gramos (g)"]
            totals = editor_df[["kcal_g", "carb_g", "prot_g", "fat_g"]].multiply(grams, axis=0).sum()
            r = {
                "nombre": recipe_name or f"Receta {len(st.session_state['recipes']) + 1}",
                "tipo_dia": tipo_dia,
                "comida": meal,
                "objetivo": {"kcal": float(kcal_target), "carb": float(c_target), "prot": float(p_target), "fat": float(f_target)},
                "resultado": {
                    "kcal": float(totals["kcal_g"]),
                    "carb": float(totals["carb_g"]),
                    "prot": float(totals["prot_g"]),
                    "fat": float(totals["fat_g"]),
                },
                "ingredientes": [
                    {"producto": editor_df.loc[i, "Producto"], "gramos": float(editor_df.loc[i, "Gramos (g)"])}
                    for i in range(len(editor_df))
                ],
            }
            st.session_state["recipes"].append(r)
            st.success("Receta guardada en la sesión.")

# =============================
# Recetas guardadas + exportación
# =============================

st.markdown("## Recetas guardadas (esta sesión)")
recipes = st.session_state.get("recipes", [])
if not recipes:
    st.caption("Aún no hay recetas guardadas.")
else:
    for day in ["Alto", "Medio", "Bajo"]:
        group = [r for r in recipes if r["tipo_dia"] == day]
        if not group:
            continue
        st.markdown(f"### {day}")
        for r in group:
            with st.expander(f"🍽️ {r['nombre']} · {r['comida']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Objetivo**")
                    st.write(f"{r['objetivo']['kcal']:.0f} kcal | C:{r['objetivo']['carb']:.0f} g · P:{r['objetivo']['prot']:.0f} g · G:{r['objetivo']['fat']:.0f} g")
                with col2:
                    st.write("**Resultado**")
                    st.write(f"{r['resultado']['kcal']:.0f} kcal | C:{r['resultado']['carb']:.0f} g · P:{r['resultado']['prot']:.0f} g · G:{r['resultado']['fat']:.0f} g")

                # Detalle por ingrediente
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
                        "Producto": ing["producto"],
                        "Gramos (g)": round(g, 1),
                        "Carbohidratos (g)": None if pd.isna(carb) else round(carb, 1),
                        "Proteína (g)": None if pd.isna(prot) else round(prot, 1),
                        "Grasa (g)": None if pd.isna(fat) else round(fat, 1),
                        "kcal": None if pd.isna(kcal) else round(kcal, 0),
                    })
                df_ing = pd.DataFrame(ing_rows)
                st.dataframe(df_ing, hide_index=True, use_container_width=True)

                # Exportar detalle a Excel (por receta)
                buf_recipe = BytesIO()
                with pd.ExcelWriter(buf_recipe, engine="openpyxl") as writer:
                    df_ing.to_excel(writer, index=False, sheet_name=r["nombre"][:31])
                buf_recipe.seek(0)
                st.download_button(
                    "Descargar receta (Excel, detalle por ingrediente)",
                    data=buf_recipe,
                    file_name=f"{r['nombre'].replace(' ', '_')}_detalle.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.caption(
                    f"Totales receta → kcal: {r['resultado']['kcal']:.0f} · "
                    f"C: {r['resultado']['carb']:.0f} g · "
                    f"P: {r['resultado']['prot']:.0f} g · "
                    f"G: {r['resultado']['fat']:.0f} g"
                )

    # Exportar todas las recetas en un solo Excel
    st.markdown("\n#### Exportar todas las recetas")
    buf_all = BytesIO()
    with pd.ExcelWriter(buf_all, engine="openpyxl") as writer:
        summary_rows = []
        for r in recipes:
            summary_rows.append({
                "Nombre": r["nombre"],
                "Tipo de día": r["tipo_dia"],
                "Comida": r["comida"],
                "kcal_obj": r["objetivo"]["kcal"],
                "carb_obj": r["objetivo"]["carb"],
                "prot_obj": r["objetivo"]["prot"],
                "fat_obj": r["objetivo"]["fat"],
                "kcal_res": r["resultado"]["kcal"],
                "carb_res": r["resultado"]["carb"],
                "prot_res": r["resultado"]["prot"],
                "fat_res": r["resultado"]["fat"],
            })
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Resumen")

        for r in recipes:
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
                    "Producto": ing["producto"],
                    "Gramos (g)": round(g, 1),
                    "Carbohidratos (g)": None if pd.isna(carb) else round(carb, 1),
                    "Proteína (g)": None if pd.isna(prot) else round(prot, 1),
                    "Grasa (g)": None if pd.isna(fat) else round(fat, 1),
                    "kcal": None if pd.isna(kcal) else round(kcal, 0),
                })
            pd.DataFrame(ing_rows).to_excel(writer, index=False, sheet_name=r["nombre"][:31])

    buf_all.seek(0)
    st.download_button(
        "Descargar TODAS las recetas (Excel)",
        data=buf_all,
        file_name="recetas_sesion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown(
    """
    ---
    **Nota**: Los datos y recetas se guardan solo durante la sesión en el navegador.
    Si quieres persistencia entre sesiones (guardar en disco o base de datos), puedo añadirlo fácilmente.
    """
)
