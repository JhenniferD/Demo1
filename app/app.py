import pandas as pd
import streamlit as st
from modelado_temporal import mostrar_series_tiempo, mostrar_modelado_forecast, bandas_confianza
from vista_general import mostrar_topn_mapa
from clustering_pca import mostrar_clustering_pca
from calidad_datos import mostrar_calidad_datos

# configuración básica
st.set_page_config(page_title="COVID-19 JHU – Métricas y Análisis",layout="wide")
st.title("COVID-19 (JHU) Dashboard")
st.caption("Fuente: Johns Hopkins CSSE – Daily Report 2022-04-18")

# Cargar los datos
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(url)
    df['Last_Update'] = pd.to_datetime(df['Last_Update'])  # Convertir a tipo datetime
    return df

# Cargar los datos
df = load_data()
df['Confirmed'] = df['Confirmed'].fillna(0).astype(int)
df['Deaths'] = df['Deaths'].fillna(0).astype(int)

# Sidebar con filtros
st.sidebar.header("Filtros")

# Por rango de fechas
if "Last_Update" in df.columns:
    df["Last_Update"] = pd.to_datetime(df["Last_Update"], errors="coerce")
    min_date = df["Last_Update"].min()
    max_date = df["Last_Update"].max()
    date_range = st.sidebar.date_input("Rango de fechas", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(df["Last_Update"].dt.date >= date_range[0]) &
                (df["Last_Update"].dt.date <= date_range[1])]

# Por países
paises = df["Country_Region"].unique()
paises_sel = st.sidebar.multiselect("Selecciona países", options=paises, default=paises[:5])
df = df[df["Country_Region"].isin(paises_sel)]

# Filtro por provincias/estados
if "Province_State" in df.columns:
    provincias = df["Province_State"].dropna().unique()
    provincias_sel = st.sidebar.multiselect("Selecciona provincias/estados", options=provincias)
    if provincias_sel:
        df = df[df["Province_State"].isin(provincias_sel)]

# Filtro por umbral de confirmados
umbral_conf = st.sidebar.slider("Umbral mínimo de confirmados", 0, int(df["Confirmed"].max()), 1000)
df = df[df["Confirmed"] >= umbral_conf]

# Filtro por carga de población
if "Population" in df.columns:
    pop_min, pop_max = st.sidebar.slider("Rango de población", 
                                         int(df["Population"].min()), 
                                         int(df["Population"].max()), 
                                         (int(df["Population"].min()), int(df["Population"].max())))
    df = df[(df["Population"] >= pop_min) & (df["Population"] <= pop_max)]

#KPIs principales
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
st.subheader("📈 KPIs Principales")
st.dataframe(grouped)

# Definición de las pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Vista General",
    "📈 Estadística Avanzada",
    "📈 Modelado temporal",
    "📊 Clustering y PCA",
    "🔎 Calidad de datos"
])

# ==========================
# Contenido de las pestañas
# ==========================

#Vista general
with tab1:
    st.header("📂 Vista General")
    mostrar_topn_mapa(df)
#Estadística
with tab2:
    st.header("📈 Estadística Avanzada")
    st.write("Aquí se calcularán las métricas clave por país (Confirmados, Fallecidos, CFR, tasas por 100k).")

#Modelado temporal
with tab3:
    st.header("🧪 Modelado temporal")
    # === 3.1 Generación de Series de Tiempo con Suavizado de 7 Días ===
    mostrar_series_tiempo(df)
    mostrar_modelado_forecast(url)
    #==bandas_confianza(df)
#Clusters
with tab4:
    st.header("📊 Clustering y PCA")
    st.write("Aquí se construirá el clustering de países con K-means y se mostrarán los grupos.")
    mostrar_clustering_pca(df)

#Calidad de datos
with tab5:
    st.header("🔎 Calidad de datos")
    st.write("Aquí se reducirá la dimensionalidad con PCA y se graficarán los componentes principales.")
    mostrar_calidad_datos(df)