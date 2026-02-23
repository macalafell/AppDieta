# AppDieta · Meal Planner

Aplicación en **Streamlit** para planificar objetivos nutricionales diarios y construir recetas por comida.

## Qué hace

- Calcula kcal basal con **Harris-Benedict revisada** (hombre/mujer) o permite introducirla manualmente.
- Define consumo diario por tipo de día (alta/media/baja) con:
  - modo multiplicador (prioritario), o
  - modo kcal extra manual.
- Calcula macros diarios por kg (proteína/grasa) y carbohidratos automáticos.
- Aplica ajuste del objetivo diario entre **-20% y +20%**.
- Reparte macros por comida (desayuno/comida/merienda/cena).
- Permite seleccionar alimentos y calcular gramos para aproximar los objetivos.
- Muestra desviación respecto al objetivo y exporta recetas en **CSV**.

---

## Requisitos

- Python 3.10+
- pip

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
```

## Ejecutar

```bash
streamlit run macro_app.py
```

---

## Archivo de alimentos (lo aportas tú)

La app lee un archivo local (`.csv` o `.xlsx`) desde la ruta que indiques en la barra lateral (`Ruta local CSV/XLSX`).

### CSV recomendado

Cabeceras esperadas (por 100g):

```csv
Alimento,Kcal,P,G,CH
Arroz blanco crudo,362,7.00,0.60,87.60
Atún al natural,116,26.00,1.00,0.00
Tomate,18,0.90,0.20,3.90
```

> Notas:
> - Los valores son por **100 gramos**.
> - Se aceptan coma o punto decimal.
> - La app normaliza nombres de columnas frecuentes automáticamente.

---

## Estructura mínima para GitHub

- `macro_app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `alimentos_template.csv` (opcional, ejemplo)

Si quieres usar tu propio CSV: colócalo en la carpeta del proyecto (por ejemplo `alimentos.csv`) y en la app deja esa misma ruta en el campo lateral `Ruta local CSV/XLSX`.
