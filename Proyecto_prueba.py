import streamlit as st
import pandas as pd
import numpy as np

url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)  
st.write("DataFrame:")
st.dataframe(df)

# Estadísticas básicas
st.subheader("Descripcion de los datos")
st.write(df.describe())

# Agrupar por país
grouped = df.groupby("Country_Region", as_index=False).agg({
    "Confirmed": "sum",
    "Deaths": "sum"
})

# Calcular CFR (muertes / confirmados)
grouped["CFR"] = (grouped["Deaths"] / grouped["Confirmed"]) * 100

#calcular un promedio por país:
incident_rate = df.groupby("Country_Region")["Incident_Rate"].mean().reset_index()
grouped = grouped.merge(incident_rate, on="Country_Region")

# Renombrar columnas
grouped = grouped.rename(columns={
    "Country_Region": "Pais",
    "Confirmed": "Confirmados",
    "Deaths": "Fallecidos",
    "CFR": "CFR (%)",
    "Incident_Rate": "Tasa casos por 100k (Incident_Rate)"
})

# Mostrar resultados
st.subheader("Metricas por pais")
st.dataframe(grouped)
