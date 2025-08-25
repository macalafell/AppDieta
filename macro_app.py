import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO

st.set_page_config(page_title="APP DIETA · Planificador de platos", layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
def mifflin_st_jeor_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    sex = sex.lower().strip()
    if sex.startswith("m") or sex == "hombre":
        return 10*weight_kg + 6.25*height_cm - 5*age + 5
    else:
        return 10*weight_kg + 6.25*height_cm - 5*age - 161

def load_foods(xls_file) -> pd.DataFrame:
    # Admite: Hoja 'Todos' o varias hojas; busca columnas esperadas y normaliza a valores por gramo
    def normalize(df):
        cols = {c.lower().strip(): c for c in df.columns}

        # kcal por gramo
        if "energía (kcal/g)" in cols or "energia (kcal/g)" in cols or "calorías (kcal/g)" in cols or "calorias (kcal/g)" in cols:
            kcal_g = df[ cols.get("energía (kcal/g)", cols.get("energia (kcal/g)", cols.get("calorías (kcal/g)", cols.get("calorias (kcal/g)")))) ]
        elif "energía (kcal/100g)" in cols or "energia (kcal/100g)" in cols or "calorías (kcal/100g)" in cols or "calorias (kcal/100g)" in cols:
            kcal_g = df[ cols.get("energía (kcal/100g)", cols.get("energia (kcal/100g)", cols.get("calorías (kcal/100g)", cols.get("calorias (kcal/100g)")))) ] / 100.0
        else:
            kcal_g = None

        # carb por gramo
        if "carbohidratos (g/g)" in cols:
            carb_g = df[ cols["carbohidratos (g/g)"] ]
        elif "carbohidratos (g/100g)" in cols:
            carb_g = df[ cols["carbohidratos (g/100g)"] ] / 100.0
        else:
            carb_g = None

        # prot por gramo
        if "proteínas (g/g)" in cols or "proteinas (g/g)" in cols:
            prot_g = df[ cols.get("proteínas (g/g)", cols.get("proteinas (g/g)")) ]
        elif "proteínas (g/100g)" in cols or "proteinas (g/100g)" in cols:
            prot_g = df[ cols.get("proteínas (g/100g)", cols.get("proteinas (g/100g)")) ] / 100.0
        else:
            prot_g = None

        # fat por gramo
        if "grasas (g/g)" in cols:
            fat_g = df[ cols["grasas (g/g)"] ]
        elif "grasas (g/100g)" in cols:
            fat_g = df[ cols["grasas (g/100g)"] ] / 100.0
        else:
            fat_g = None

        name_col = cols.get("producto") or cols.get("alimento") or "Producto"
        brand_col = cols.get("marca") or "Marca"
        cat_col = cols.get("categoría") or cols.get("categoria") or "Categoría"
        subcat_col = cols.get("subcategoría") or cols.get("subcategoria") or "Subcategoría"

        clean = pd.DataFrame({
            "Producto": df[name_col].astype(str),
            "Marca": df[brand_col] if brand_col in df.columns else "",
            "Categoría": df[cat_col] if cat_col in df.columns else "",
            "Subcategoría": df[subcat_col] if subcat_col in df.columns else "",
            "kcal_g": kcal_g.astype(float) if kcal_g is not None else np.nan,
            "carb_g": carb_g.astype(float) if carb_g is not None else np.nan,
            "prot_g": prot_g.astype(float) if prot_g is not None else np.nan,
            "fat_g": fat_g.astype(float) if fat_g is not None else np.nan,
        })
        clean = clean.dropna(subset=["kcal_g", "carb_g", "prot_g", "fat_g"])
        return clean

    try:
        xls = pd.ExcelFile(xls_file)
        if "Todos" in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name="Todos")
            return normalize(df)
        dfs = []
        for s in xls.sheet_names:
            dfs.append(normalize(pd.read_excel(xls, sheet_name=s)))
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")
        return pd.DataFrame()

def nnls_iterative(A, b, max_iter=6):
    # Solve A x ≈ b with x >= 0 by iterative least squares (sin SciPy)
    used = np.arange(A.shape[1])
    x = np.maximum(0, np.linalg.lstsq(A, b, rcond=None)[0])
    for _ in range(max_iter):
        neg = x < 0
        if not neg.any():
            break
        keep = ~neg
        used = used[keep]
        if used.size == 0:
            return np.zeros(A.shape[1])
        A2 = A[:, keep]
        x2 = np.maximum(0, np.linalg.lstsq(A2, b, rcond=None)[0])
        x = np.zeros_like(x)
        x[keep] = x2
    return np.maximum(0, x)

def kcal_from_macros(carb_g, prot_g, fat_g):
    return carb_g*4 + prot_g*4 + fat_g*9

# -----------------------------
# Sidebar: Datos del usuario y parámetros
# -----------------------------
st.sidebar.header("Perfil y parámetros")

sex = st.sidebar.selectbox("Sexo", ["Hombre", "Mujer"])
weight = st.sidebar.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.5)
height = st.sidebar.number_input("Altura (cm)", min_value=120.0, max_value=230.0, value=178.0, step=0.5)
age = st.sidebar.number_input("Edad (años)", min_value=14, max_value=100, value=35, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Multiplicadores por tipo de día")
mult_alto = st.sidebar.number_input("Alto", value=1.60, step=0.01, format="%.2f")
mult_medio = st.sidebar.number_input("Medio", value=1.55, step=0.01, format="%.2f")
mult_bajo = st.sidebar.number_input("Bajo", value=1.50, step=0.01, format="%.2f")

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

# === Carbohidratos (g/día) calculados por tipo de día — NO editable ===
st.sidebar.markdown("---")
st.sidebar.subheader("Carbohidratos (g/día) calculados")

def carbs_for_day(mult, p_gkg, f_gkg, adj_pct_value):
    tdee_x = mifflin_st_jeor_bmr(sex, weight, height, age) * mult
    tdee_x = tdee_x * (1 + adj_pct_value/100.0)
    p_day_x = p_gkg * weight
    f_day_x = f_gkg * weight
    c_day_x = max(0.0, (tdee_x - (p_day_x*4 + f_day_x*9)) / 4.0)
    return round(float(c_day_x), 1)

carbs_alto  = carbs_for_day(mult_alto,  p_alto,  g_alto,  adj_pct)
carbs_medio = carbs_for_day(mult_medio, p_medio, g_medio, adj_pct)
carbs_bajo  = carbs_for_day(mult_bajo,  p_bajo,  g_bajo,  adj_pct)

st.sidebar.caption("Día ALTO")
st.sidebar.number_input(
    "Carbohidratos (g/día) - ALTO",
    value=carbs_alto, step=0.1, format="%.1f",
    disabled=True, key="c_alto_readonly",
)
st.sidebar.caption("Día MEDIO")
st.sidebar.number_input(
    "Carbohidratos (g/día) - MEDIO",
    value=carbs_medio, step=0.1, format="%.1f",
    disabled=True, key="c_medio_readonly",
)
st.sidebar.caption("Día BAJO")
st.sidebar.number_input(
    "Carbohidratos (g/día) - BAJO",
    value=carbs_bajo, step=0.1, format="%.1f",
    disabled=True, key="c_bajo_readonly",
)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Sube tu Excel de alimentos (opcional)", type=["xlsx"])

# Cargar datos
if uploaded is not None:
    foods = load_foods(uploaded)
else:
    default_path = "alimentos_800_especificos.xlsx"
    foods = load_foods(default_path) if os.path.exists(default_path) else pd.DataFrame()

# -----------------------------
# Cálculos diarios
# -----------------------------
bmr = mifflin_st_jeor_bmr(sex, weight, height, age)
tipo_dia = st.selectbox("Tipo de día", ["Alto", "Medio", "Bajo"])

mult = {"Alto": mult_alto, "Medio": mult_medio, "Bajo": mult_bajo}[tipo_dia]
tdee = bmr * mult
tdee = tdee * (1 + adj_pct/100.0)

# Macros diarios objetivo
p_day = {"Alto": p_alto, "Medio": p_medio, "Bajo": p_bajo}[tipo_dia] * weight
g_day = {"Alto": g_alto, "Medio": g_medio, "Bajo": g_bajo}[tipo_dia] * weight
kcal_from_p_f = p_day*4 + g_day*9
c_day = max(0.0, (tdee - kcal_from_p_f) / 4.0)

left, right = st.columns([1,1])
with left:
    st.metric("BMR (kcal/día)", f"{bmr:.0f}")
    st.metric("TDEE (kcal/día)", f"{tdee:.0f}")
with right:
    st.metric("Proteína (g/día)", f"{p_day:.0f}")
    st.metric("Grasa (g/día)", f"{g_day:.0f}")
    st.metric("Carbohidratos (g/día)", f"{c_day:.0f}")

# -----------------------------
# Reparto por comida (editable)
# -----------------------------
st.markdown("### Reparto por comida")
defaults = {
    "Desayuno": {"prot": 0.10, "fat": 0.10, "carb": 0.27},
    "Comida":   {"prot": 0.39, "fat": 0.40, "carb": 0.26},
    "Merienda": {"prot": 0.08, "fat": 0.06, "carb": 0.17},
    "Cena":     {"prot": 0.43, "fat": 0.44, "carb": 0.30},
}
with st.expander("Editar porcentajes de reparto por comida (proporción del día)"):
    for key in defaults:
        st.write(f"**{key}**")
        defaults[key]["prot"] = st.number_input(f"Proteína ({key})", value=float(defaults[key]["prot"]), step=0.01, format="%.2f", key=f"p_{key}")
        defaults[key]["fat"]  = st.number_input(f"Grasa ({key})", value=float(defaults[key]["fat"]), step=0.01, format="%.2f", key=f"g_{key}")
        defaults[key]["carb"] = st.number_input(f"Hidratos ({key})", value=float(defaults[key]["carb"]), step=0.01, format="%.2f", key=f"c_{key}")

# -----------------------------
# NUEVO: Resumen de macros por comida + exportar Excel
# -----------------------------
st.markdown("### Resumen de macros por comida")

def meal_targets(meal_name, perc_dict):
    p_t = p_day * perc_dict["prot"]
    f_t = g_day * perc_dict["fat"]
    c_t = c_day * perc_dict["carb"]
    kcal_t = c_t*4 + p_t*4 + f_t*9
    return {
        "Comida": meal_name,
        "kcal": round(kcal_t, 0),
        "Carbohidratos (g)": round(c_t, 1),
        "Proteína (g)": round(p_t, 1),
        "Grasa (g)": round(f_t, 1),
    }

meals_summary = pd.DataFrame([
    meal_targets("Desayuno", defaults["Desayuno"]),
    meal_targets("Comida",   defaults["Comida"]),
    meal_targets("Merienda", defaults["Merienda"]),
    meal_targets("Cena",     defaults["Cena"]),
])

# ➕ Fila TOTAL
totals_row = {
    "Comida": "TOTAL",
    "kcal": round(meals_summary["kcal"].sum(), 0),
    "Carbohidratos (g)": round(meals_summary["Carbohidratos (g)"].sum(), 1),
    "Proteína (g)": round(meals_summary["Proteína (g)"].sum(), 1),
    "Grasa (g)": round(meals_summary["Grasa (g)"].sum(), 1),
}
meals_summary_total = pd.concat([meals_summary, pd.DataFrame([totals_row])], ignore_index=True)

# Mostrar con totales
st.dataframe(meals_summary_total, use_container_width=True)

# Exportar con totales
buf_meals = BytesIO()
with pd.ExcelWriter(buf_meals, engine="openpyxl") as writer:
    meals_summary_total.to_excel(writer, index=False, sheet_name="Macros por comida")
buf_meals.seek(0)
st.download_button(
    "Descargar resumen por comida (Excel)",
    data=buf_meals,
    file_name="resumen_macros_por_comida.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


# -----------------------------
# Objetivos por comida seleccionada
# -----------------------------
meal = st.selectbox("Comida", ["Desayuno", "Comida", "Merienda", "Cena"])
perc = defaults[meal]
p_target = p_day * perc["prot"]
f_target = g_day * perc["fat"]
c_target = c_day * perc["carb"]
kcal_target = c_target*4 + p_target*4 + f_target*9
st.info(f"Objetivo para {meal} → {kcal_target:.0f} kcal | Prot: {p_target:.0f} g | Grasa: {f_target:.0f} g | Hidratos: {c_target:.0f} g")

# -----------------------------
# Creador de receta
# -----------------------------
st.markdown("### Creador de receta")
if foods.empty:
    st.warning("Primero sube o coloca en la carpeta un Excel de alimentos (alimentos_800_especificos.xlsx).")
else:
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        sel_cat = st.selectbox("Filtrar por categoría", ["(Todas)"] + sorted(foods["Categoría"].astype(str).unique().tolist()))
    with fcol2:
        sel_sub = st.selectbox("Filtrar por subcategoría", ["(Todas)"] + sorted(foods["Subcategoría"].astype(str).unique().tolist()))
    with fcol3:
        search = st.text_input("Buscar por nombre/marca contiene…", "")

    df_view = foods.copy()
    if sel_cat != "(Todas)":
        df_view = df_view[df_view["Categoría"] == sel_cat]
    if sel_sub != "(Todas)":
        df_view = df_view[df_view["Subcategoría"] == sel_sub]
    if search.strip():
        s = search.strip().lower()
        df_view = df_view[df_view["Producto"].str.lower().str.contains(s) | df_view["Marca"].astype(str).str.lower().str.contains(s)]

    st.dataframe(df_view[["Producto","Marca","kcal/g","carb/g","prot/g","fat/g"]].rename(columns={
        "kcal_g":"kcal/g","carb_g":"carb/g","prot_g":"prot/g","fat_g":"fat/g"
    }), use_container_width=True, height=300)

    choices = st.multiselect("Elige alimentos para la receta", df_view["Producto"].tolist())
    selected = df_view[df_view["Producto"].isin(choices)].reset_index(drop=True)

    if not selected.empty:
        grams = []
        st.write("Introduce gramos (puedes dejar a 0 y usar el ajuste automático):")
        for i, row in selected.iterrows():
            grams.append(st.number_input(f"{row['Producto']} (g)", min_value=0.0, value=0.0, step=5.0, key=f"g_{i}"))
        grams = np.array(grams)

        if st.button("Ajustar gramos automáticamente para cuadrar macros"):
            A = selected[["kcal_g","carb_g","prot_g","fat_g"]].to_numpy().T
            b = np.array([kcal_target, c_target, p_target, f_target], dtype=float)
            x = nnls_iterative(A, b, max_iter=6)
            grams = x
            st.success("Gramos ajustados.")

        totals = selected[["kcal_g","carb_g","prot_g","fat_g"]].multiply(grams, axis=0).sum()
        kcal_tot = float(totals["kcal_g"])
        carb_tot = float(totals["carb_g"])
        prot_tot = float(totals["prot_g"])
        fat_tot = float(totals["fat_g"])

        colA, colB, colC, colD = st.columns(4)
        colA.metric("kcal", f"{kcal_tot:.0f}", delta=f"{kcal_tot-kcal_target:+.0f}")
        colB.metric("Carbohidratos (g)", f"{carb_tot:.0f}", delta=f"{carb_tot-c_target:+.0f}")
        colC.metric("Proteínas (g)", f"{prot_tot:.0f}", delta=f"{prot_tot-p_target:+.0f}")
        colD.metric("Grasas (g)", f"{fat_tot:.0f}", delta=f"{fat_tot-f_target:+.0f}")

        # NUEVO: Detalle actual (ingrediente -> gramos de producto y de macros) antes de guardar
        current_rows = []
        for i, row in selected.iterrows():
            g = float(grams[i])
            current_rows.append({
                "Producto": row["Producto"],
                "Gramos (g)": round(g, 1),
                "Carbohidratos (g)": round(row["carb_g"]*g, 1),
                "Proteína (g)": round(row["prot_g"]*g, 1),
                "Grasa (g)": round(row["fat_g"]*g, 1),
                "kcal": round(row["kcal_g"]*g, 0),
            })
        if current_rows:
            df_curr = pd.DataFrame(current_rows)
            st.write("**Detalle actual de la receta (antes de guardar)**")
            st.dataframe(df_curr, hide_index=True, use_container_width=True)

        st.markdown("---")
        recipe_name = st.text_input("Nombre de la receta")
        if "recipes" not in st.session_state:
            st.session_state["recipes"] = []
        if st.button("Guardar receta"):
            st.session_state["recipes"].append({
                "nombre": recipe_name or f"Receta {len(st.session_state['recipes'])+1}",
                "tipo_dia": tipo_dia,
                "comida": meal,
                "objetivo": {
                    "kcal": float(kcal_target),
                    "carb": float(c_target),
                    "prot": float(p_target),
                    "fat": float(f_target),
                },
                "resultado": {
                    "kcal": kcal_tot,
                    "carb": carb_tot,
                    "prot": prot_tot,
                    "fat": fat_tot,
                },
                "ingredientes": [
                    {"producto": selected.loc[i,"Producto"], "gramos": float(grams[i])}
                    for i in range(len(selected))
                ]
            })
            st.success("Receta guardada en la sesión.")

# -----------------------------
# Recetas guardadas (con detalle y exportación)
# -----------------------------
st.markdown("## Recetas guardadas (esta sesión)")
recipes = st.session_state.get("recipes", [])
if not recipes:
    st.caption("Aún no hay recetas guardadas.")
else:
    for day in ["Alto","Medio","Bajo"]:
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

                # Detalle por ingrediente con macros y kcal
                st.write("**Ingredientes: gramos y macros**")
                ing_rows = []
                for ing in r["ingredientes"]:
                    row = foods[foods["Producto"] == ing["producto"]].head(1)
                    if not row.empty:
                        g = float(ing["gramos"])
                        kcal = float(row["kcal_g"].iloc[0]) * g
                        carb = float(row["carb_g"].iloc[0]) * g
                        prot = float(row["prot_g"].iloc[0]) * g
                        fat  = float(row["fat_g"].iloc[0])  * g
                        ing_rows.append({
                            "Producto": ing["producto"],
                            "Gramos (g)": round(g, 1),
                            "Carbohidratos (g)": round(carb, 1),
                            "Proteína (g)": round(prot, 1),
                            "Grasa (g)": round(fat, 1),
                            "kcal": round(kcal, 0),
                        })
                    else:
                        ing_rows.append({
                            "Producto": ing["producto"],
                            "Gramos (g)": round(float(ing["gramos"]), 1),
                            "Carbohidratos (g)": None,
                            "Proteína (g)": None,
                            "Grasa (g)": None,
                            "kcal": None,
                        })
                df_ing = pd.DataFrame(ing_rows)
                st.dataframe(df_ing, hide_index=True, use_container_width=True)

                # Exportar detalle a Excel
                buf_recipe = BytesIO()
                with pd.ExcelWriter(buf_recipe, engine="openpyxl") as writer:
                    df_ing.to_excel(writer, index=False, sheet_name=r["nombre"][:31])
                buf_recipe.seek(0)
                st.download_button(
                    "Descargar receta (Excel, detalle por ingrediente)",
                    data=buf_recipe,
                    file_name=f"{r['nombre'].replace(' ','_')}_detalle.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.caption(
                    f"Totales receta → kcal: {r['resultado']['kcal']:.0f} · "
                    f"C: {r['resultado']['carb']:.0f} g · "
                    f"P: {r['resultado']['prot']:.0f} g · "
                    f"G: {r['resultado']['fat']:.0f} g"
                )

st.markdown(
    """
    ---
    **Nota**: Los datos y recetas se guardan solo durante la sesión en el navegador.
    Si quieres persistencia entre sesiones (guardar en disco), pídemelo y lo añadimos
    (por ejemplo, guardando en un archivo Excel o CSV).
    """
)
