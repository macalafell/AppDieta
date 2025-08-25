
import streamlit as st
import pandas as pd
import numpy as np
import os

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
    # Admite: Hoja 'Todos' o varias hojas; buscar columnas esperadas
    def normalize(df):
        cols = {c.lower().strip(): c for c in df.columns}
        # Detecta per-gramo o por 100 g y crea columnas por gramo
        kcal_g = None
        if "energía (kcal/g)" in cols or "energia (kcal/g)" in cols or "calorías (kcal/g)" in cols or "calorias (kcal/g)" in cols:
            kcal_g = df[ cols.get("energía (kcal/g)", cols.get("energia (kcal/g)", cols.get("calorías (kcal/g)", cols.get("calorias (kcal/g)")))) ]
        elif "energía (kcal/100g)" in cols or "energia (kcal/100g)" in cols or "calorías (kcal/100g)" in cols or "calorias (kcal/100g)" in cols:
            kcal_g = df[ cols.get("energía (kcal/100g)", cols.get("energia (kcal/100g)", cols.get("calorías (kcal/100g)", cols.get("calorias (kcal/100g)")))) ] / 100.0
        
        carbs_g = None
        if "carbohidratos (g/g)" in cols:
            carbs_g = df[ cols["carbohidratos (g/g)"] ]
        elif "carbohidratos (g/100g)" in cols:
            carbs_g = df[ cols["carbohidratos (g/100g)"] ] / 100.0
        
        prot_g = None
        if "proteínas (g/g)" in cols or "proteinas (g/g)" in cols:
            prot_g = df[ cols.get("proteínas (g/g)", cols.get("proteinas (g/g)")) ]
        elif "proteínas (g/100g)" in cols or "proteinas (g/100g)" in cols:
            prot_g = df[ cols.get("proteínas (g/100g)", cols.get("proteinas (g/100g)")) ] / 100.0
        
        fat_g = None
        if "grasas (g/g)" in cols:
            fat_g = df[ cols["grasas (g/g)"] ]
        elif "grasas (g/100g)" in cols:
            fat_g = df[ cols["grasas (g/100g)"] ] / 100.0
        
        # Nombres
        name_col = cols.get("producto") or cols.get("alimento") or "Producto"
        brand_col = cols.get("marca") or "Marca"
        cat_col = cols.get("categoría") or cols.get("categoria") or "Categoría"
        subcat_col = cols.get("subcategoría") or cols.get("subcategoria") or "Subcategoría"
        
        clean = pd.DataFrame({
            "Producto": df[name_col].astype(str),
            "Marca": df[brand_col] if brand_col in df.columns else "",
            "Categoría": df[cat_col] if cat_col in df.columns else "",
            "Subcategoría": df[subcat_col] if subcat_col in df.columns else "",
            "kcal_g": kcal_g.astype(float),
            "carb_g": carbs_g.astype(float),
            "prot_g": prot_g.astype(float),
            "fat_g": fat_g.astype(float),
        })
        clean = clean.dropna(subset=["kcal_g", "carb_g", "prot_g", "fat_g"])
        return clean

    try:
        xls = pd.ExcelFile(xls_file)
        sheets = xls.sheet_names
        # Si existe 'Todos', priorizar
        if "Todos" in sheets:
            df = pd.read_excel(xls, sheet_name="Todos")
            return normalize(df)
        # si no, concatenar todas
        dfs = []
        for s in sheets:
            dfs.append(normalize(pd.read_excel(xls, sheet_name=s)))
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")
        return pd.DataFrame()

def nnls_iterative(A, b, max_iter=5):
    """Solve A·x ≈ b with x≥0 by iterative least squares."""
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
# Alto
st.sidebar.caption("Día ALTO")
p_alto = st.sidebar.number_input("Proteína (g/kg) - ALTO", value=1.4, step=0.1)
g_alto = st.sidebar.number_input("Grasa (g/kg) - ALTO", value=0.7, step=0.1)
# Medio
st.sidebar.caption("Día MEDIO")
p_medio = st.sidebar.number_input("Proteína (g/kg) - MEDIO", value=1.7, step=0.1)
g_medio = st.sidebar.number_input("Grasa (g/kg) - MEDIO", value=1.1, step=0.1)
# Bajo
st.sidebar.caption("Día BAJO")
p_bajo = st.sidebar.number_input("Proteína (g/kg) - BAJO", value=2.0, step=0.1)
g_bajo = st.sidebar.number_input("Grasa (g/kg) - BAJO", value=1.5, step=0.1)

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

# Cargar datos
if uploaded is not None:
    foods = load_foods(uploaded)
else:
    # Intentar encontrar un archivo por defecto en la carpeta
    default_path = "alimentos_800_especificos.xlsx"
    if os.path.exists(default_path):
        foods = load_foods(default_path)
    else:
        foods = pd.DataFrame()

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
# Carbohidratos = resto de kcal
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
# Reparto por comida
# -----------------------------
st.markdown("### Reparto por comida")
meal = st.selectbox("Comida", ["Desayuno", "Comida", "Merienda", "Cena"])

# Porcentajes por defecto (del TOTAL DIARIO) para cada comida
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

perc = defaults[meal]
p_target = p_day * perc["prot"]
f_target = g_day * perc["fat"]
c_target = c_day * perc["carb"]
kcal_target = c_target*4 + p_target*4 + f_target*9

st.info(f"**Objetivo para {meal}** → {kcal_target:.0f} kcal | Prot: {p_target:.0f} g | Grasa: {f_target:.0f} g | Hidratos: {c_target:.0f} g")

# -----------------------------
# Constructor de recetas
# -----------------------------
st.markdown("### Creador de receta")
if foods.empty:
    st.warning("Primero sube o coloca en la carpeta un Excel de alimentos (alimentos_800_especificos.xlsx).")
else:
    # Filtros rápidos
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

    st.dataframe(df_view[["Producto","Marca","kcal_g","carb_g","prot_g","fat_g"]].rename(columns={
        "kcal_g":"kcal/g","carb_g":"carb/g","prot_g":"prot/g","fat_g":"fat/g"
    }), use_container_width=True, height=300)

    # Selección de alimentos
    choices = st.multiselect("Elige alimentos para la receta", df_view["Producto"].tolist())
    selected = df_view[df_view["Producto"].isin(choices)].reset_index(drop=True)

    if not selected.empty:
        grams = []
        st.write("Introduce gramos (puedes dejar a 0 y usar el ajuste automático):")
        for i, row in selected.iterrows():
            grams.append(st.number_input(f"{row['Producto']} (g)", min_value=0.0, value=0.0, step=5.0, key=f"g_{i}"))
        grams = np.array(grams)

        # Botón autoajuste
        if st.button("Ajustar gramos automáticamente para cuadrar macros"):
            # Matriz A: filas = [kcal, carb, prot, fat], columnas = alimentos (por gramo)
            A = selected[["kcal_g","carb_g","prot_g","fat_g"]].to_numpy().T  # 4 x n
            b = np.array([kcal_target, c_target, p_target, f_target], dtype=float)
            x = nnls_iterative(A, b, max_iter=6)
            grams = x
            st.success("Gramos ajustados.")
        
        # Resultados de la receta
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

        # Guardado de recetas en sesión
        st.markdown("---")
        recipe_name = st.text_input("Nombre de la receta")
        if "recipes" not in st.session_state:
            st.session_state["recipes"] = []
        if st.button("Guardar receta"):
            items = []
            for _, row in selected.iterrows():
                items.append({
                    "Producto": row["Producto"],
                    "Marca": row["Marca"],
                    "kcal_g": row["kcal_g"],
                    "carb_g": row["carb_g"],
                    "prot_g": row["prot_g"],
                    "fat_g": row["fat_g"],
                })
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
# Recetas guardadas
# -----------------------------
st.markdown("## Recetas guardadas (esta sesión)")
recipes = st.session_state.get("recipes", [])
if not recipes:
    st.caption("Aún no hay recetas guardadas.")
else:
    # Mostrar agrupadas por tipo de día
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
                st.write("**Ingredientes (g)**")
                df_ing = pd.DataFrame(r["ingredientes"])
                st.dataframe(df_ing, hide_index=True, use_container_width=True)

                # Botón para descargar la receta en CSV
                df_out = df_ing.copy()
                df_out["tipo_dia"] = r["tipo_dia"]
                df_out["comida"] = r["comida"]
                df_out["kcal_obj"] = r["objetivo"]["kcal"]
                df_out["carb_obj"] = r["objetivo"]["carb"]
                df_out["prot_obj"] = r["objetivo"]["prot"]
                df_out["fat_obj"]  = r["objetivo"]["fat"]
                df_out["kcal_real"] = r["resultado"]["kcal"]
                df_out["carb_real"] = r["resultado"]["carb"]
                df_out["prot_real"] = r["resultado"]["prot"]
                df_out["fat_real"]  = r["resultado"]["fat"]

                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar receta (CSV)", data=csv, file_name=f"{r['nombre'].replace(' ','_')}.csv", mime="text/csv")

st.markdown(
    """
    ---
    **Nota**: Los datos y recetas se guardan solo durante la sesión en el navegador.  
    Si quieres persistencia entre sesiones (guardar en disco), pídemelo y lo añadimos  
    (por ejemplo, guardando en un archivo CSV/Excel "recetas_guardadas.xlsx").
    """
)
