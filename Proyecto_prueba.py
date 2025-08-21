import streamlit as st
import pandas as pd
import numpy as np
# Título y subtítulos
st.title("Mi primera aplicación con Streamlit 🎉")
st.header("Ejemplo usando pandas y numpy")
st.subheader("Mostrando texto, tablas y gráficos")
# Texto simple con formato Markdown
st.markdown("Hola, este es un *ejemplo sencillo* de una app con Streamlit")
# Crear un DataFrame con pandas y numpy
data = pd.DataFrame(
np.random.randn(10, 3), # 10 filas, 3 columnas con datos aleatorios
columns=["Columna A", "Columna B", "Columna C"]
)
# Mostrar la tabla
st.write("Aquí tienes una tabla generada con Pandas:")
st.dataframe(data)
# Mostrar estadísticas
st.write("Resumen estadístico de los datos:")
st.write(data.describe())
# Mostrar gráfico interactivo
st.line_chart(data)
# Texto adicional con LaTeX
st.latex(r"E = mc^2")

df = pd.read_csv("Archivo.csv")   # asegúrate de que "datos.csv" esté en la misma carpeta

st.success("✅ Archivo cargado con éxito")
st.write("📊 Vista previa del DataFrame:")
st.dataframe(df)

# Estadísticas básicas
st.subheader("📈 Descripción de los datos")
st.write(df.describe())

st.write("📂 Vista previa de los datos:")
st.dataframe(df.head())

# Agrupar por país
grouped = df.groupby("Country_Region", as_index=False).agg({
    "Confirmed": "sum",
    "Deaths": "sum"
})

# Calcular CFR (muertes / confirmados)
grouped["CFR"] = (grouped["Deaths"] / grouped["Confirmed"]) * 100

# Si quieres usar la columna `Incident_Rate` (que ya es casos por 100k),
# puedes calcular un promedio por país:
incident_rate = df.groupby("Country_Region")["Incident_Rate"].mean().reset_index()
grouped = grouped.merge(incident_rate, on="Country_Region")

# Renombrar columnas para más claridad
grouped = grouped.rename(columns={
    "Country_Region": "País",
    "Confirmed": "Confirmados",
    "Deaths": "Fallecidos",
    "CFR": "CFR (%)",
    "Incident_Rate": "Tasa casos por 100k (Incident_Rate)"
})

# Mostrar resultados
st.subheader("📈 Métricas por país")
st.dataframe(grouped)

# Exportar a CSV
csv = grouped.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Descargar métricas en CSV",
    data=csv,
    file_name="metricas_covid.csv",
    mime="text/csv"
)
