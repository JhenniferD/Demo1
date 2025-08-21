import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("Archivo.csv")  

# Asegurar que la fecha esté en formato datetime
df["Last_Update"] = pd.to_datetime(df["Last_Update"])

# 1. Ordenamos por fecha (más antigua → más reciente)
df = df.sort_values("Last_Update")

# 2. Tomamos la última fecha disponible del dataset
ultima_fecha = df["Last_Update"].max()

# 3. Filtramos solo esa fecha
df_latest = df[df["Last_Update"] == ultima_fecha]

# 4. Agrupamos por país (sumando provincias)
grouped = df_latest.groupby("Country_Region", as_index=False).agg({
    "Confirmed": "sum",
    "Deaths": "sum"
})

# 5. Calculamos métricas
grouped["CFR (%)"] = (grouped["Deaths"] / grouped["Confirmed"]) * 100
# si tienes Incident_Rate (casos por 100k) ya está calculado en el CSV
# pero lo podemos promediar si viene por provincia
if "Incident_Rate" in df.columns:
    grouped["Casos por 100k"] = df_latest.groupby("Country_Region")["Incident_Rate"].mean().values

# Renombramos para claridad
grouped = grouped.rename(columns={
    "Country_Region": "País",
    "Confirmed": "Confirmados",
    "Deaths": "Fallecidos"
})

# Mostrar resultados
st.subheader("Metricas por pais")
st.dataframe(grouped)
