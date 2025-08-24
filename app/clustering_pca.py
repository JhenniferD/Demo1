import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 800
pio.kaleido.scope.default_height = 600
pio.kaleido.scope.default_scale = 1


# ==============================
# Utilidad: preparar métricas por país
# ==============================
def _prep_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un dataframe por país con:
    - Confirmados, Muertes (sumas)
    - Incident_Rate (promedio por país) como "tasa por 100k"
    - CFR (%)
    - Crecimiento 7d (si existe información para >= 8 fechas)
    """
    work = df.copy()

    # Normalizar nombres de columnas esperadas en el CSV de JHU
    cols = {c.lower(): c for c in work.columns}
    col_country = cols.get("country_region", "Country_Region")
    col_confirmed = cols.get("confirmed", "Confirmed")
    col_deaths = cols.get("deaths", "Deaths")
    col_incident = cols.get("incident_rate", "Incident_Rate")
    col_last_update = cols.get("last_update", "Last_Update")

    # Asegurar tipo datetime si existe
    if col_last_update in work.columns:
        work[col_last_update] = pd.to_datetime(work[col_last_update], errors="coerce")

    # Agregación base por país
    grouped = work.groupby(col_country, as_index=False).agg({
        col_confirmed: "sum",
        col_deaths: "sum",
    })

    # Tasas por 100k (Incident Rate) -> promedio por país (hay varias filas por provincia)
    if col_incident in work.columns:
        ir = work.groupby(col_country)[col_incident].mean().reset_index()
        grouped = grouped.merge(ir, on=col_country, how="left")
    else:
        grouped[col_incident] = np.nan

    # CFR (%)
    grouped["CFR"] = np.where(grouped[col_confirmed] > 0,
                              (grouped[col_deaths] / grouped[col_confirmed]) * 100,
                              0.0)

    # Crecimiento 7d: sólo si hay múltiples fechas en los datos
    growth_map = {}
    if col_last_update in work.columns and work[col_last_update].dt.date.nunique() >= 8:
        daily = (work
                 .groupby([col_country, work[col_last_update].dt.date])[col_confirmed]
                 .sum()
                 .reset_index(name="confirmed_day"))

        # Ordenar y calcular rolling de 7 días por país
        daily = daily.sort_values([col_country, col_last_update])
        daily["confirmed_t_7"] = (daily
                                  .groupby(col_country)["confirmed_day"]
                                  .shift(7))
        daily["growth_7d"] = (daily["confirmed_day"] - daily["confirmed_t_7"]) / daily["confirmed_t_7"]
        # Tomar el último valor disponible por país
        last_growth = (daily
                       .dropna(subset=["growth_7d"])
                       .groupby(col_country)["growth_7d"]
                       .last()
                       .reset_index())
        growth_map = dict(zip(last_growth[col_country], last_growth["growth_7d"]))

    grouped["Growth_7d"] = grouped[col_country].map(growth_map)

    # Si no se pudo calcular growth_7d (dataset de un solo día), ponemos 0 pero avisamos luego en UI.
    if grouped["Growth_7d"].isna().all():
        grouped["Growth_7d"] = 0.0
        grouped["_growth_note"] = True
    else:
        grouped["_growth_note"] = False

    # Renombrar para consistencia en UI
    grouped = grouped.rename(columns={
        col_country: "Pais",
        col_confirmed: "Confirmados",
        col_deaths: "Fallecidos",
        col_incident: "Tasa por 100k (Incident_Rate)"
    })

    return grouped


# ==============================
# 4.1 Clustering (K-means u otro)
# ==============================
def _clustering_41(data: pd.DataFrame, k: int = 4, random_state: int = 42):
    """Devuelve dataframe con etiqueta de clúster y el modelo entrenado."""
    features = data[[
        "Tasa por 100k (Incident_Rate)",
        "CFR",
        "Growth_7d"
    ]].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    clusters = km.fit_predict(X)
    result = data.copy()
    result["Cluster"] = clusters.astype(int)
    return result, km, scaler


# ==============================
# 4.2 PCA a 2 componentes y scatter
# ==============================
def _pca_42(data_with_cluster: pd.DataFrame, scaler: StandardScaler, random_state: int = 42):
    """Calcula PCA(2) sobre las features estandarizadas y agrega PC1/PC2."""
    X = scaler.transform(data_with_cluster[[
        "Tasa por 100k (Incident_Rate)",
        "CFR",
        "Growth_7d"
    ]].values)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    df_pca = data_with_cluster.copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=["Tasa por 100k (Incident_Rate)", "CFR", "Growth_7d"],
        columns=["PC1", "PC2"]
    )
    return df_pca, pca, loadings


# ==============================
# 4.3 Interpretación automática de perfiles
# ==============================
def _interpretar_43(df_clustered: pd.DataFrame):
    profile = (df_clustered
               .groupby("Cluster")[["Tasa por 100k (Incident_Rate)", "CFR", "Growth_7d",
                                    "Confirmados", "Fallecidos"]]
               .mean()
               .sort_index())

    interpretations = []
    percentiles = df_clustered[["Tasa por 100k (Incident_Rate)", "CFR", "Growth_7d"]].quantile([0.33, 0.66])

    def lvl(val, qlow, qhigh):
        if val <= qlow: return "baja"
        if val >= qhigh: return "alta"
        return "media"

    for cl, row in profile.iterrows():
        tasa = row["Tasa por 100k (Incident_Rate)"]
        cfr = row["CFR"]
        g7 = row["Growth_7d"]
        txt = (
            f"**Cluster {cl}:** Países con "
            f"tasa por 100k {lvl(tasa, percentiles.iloc[0,0], percentiles.iloc[1,0])}, "
            f"CFR {lvl(cfr, percentiles.iloc[0,1], percentiles.iloc[1,1])} y "
            f"crecimiento 7d {lvl(g7, percentiles.iloc[0,2], percentiles.iloc[1,2])}."
        )
        interpretations.append(txt)

    return profile, interpretations


# ==============================
# Punto 4 completo (UI)
# ==============================
def mostrar_clustering_pca(df: pd.DataFrame):
    """
    4. Segmentación y reducción de dimensionalidad
    4.1 Clustering K-means (tasas, CFR, crecimiento 7d)
    4.2 PCA a 2 componentes + scatter (PC1 vs PC2)
    4.3 Interpretación y perfiles por clúster
    """
    st.markdown("### 4. Segmentación y Reducción de Dimensionalidad")

    # --------- Preparación de datos
    data = _prep_country_metrics(df)

    # Parámetros en sidebar
    st.sidebar.subheader("Parámetros de Clustering (Pestaña Clustering y PCA)")
    k = st.sidebar.slider("Número de clústeres (k)", min_value=2, max_value=8, value=4, step=1)
    random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

    # ==================== 4.1
    st.markdown("#### 4.1 Clustering por país (K-means)")
    clustered, km, scaler = _clustering_41(data, k=k, random_state=random_state)
    st.dataframe(clustered[[
        "Pais", "Confirmados", "Fallecidos",
        "Tasa por 100k (Incident_Rate)", "CFR", "Growth_7d", "Cluster"
    ]].sort_values(["Cluster", "Confirmados"], ascending=[True, False]))

    if clustered["_growth_note"].any():
        st.info("**Nota:** No hay múltiples fechas en el dataset cargado, por lo que "
                "el crecimiento 7d se fijó en 0 para todos los países. Si cargas múltiples "
                "días, se calculará automáticamente por país.")
    # Descargar resultados como CSV
    csv_clusters = clustered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resultados de clustering (CSV)",
        data=csv_clusters,
        file_name="clustering_resultados.csv",
        mime="text/csv"
    )
    # ==================== 4.2
    st.markdown("#### 4.2 PCA (2 componentes) y visualización")
    df_pca, pca, loadings = _pca_42(clustered, scaler, random_state=random_state)

    fig = px.scatter(
        df_pca,
        x="PC1", y="PC2",
        color=df_pca["Cluster"].astype(str),
        hover_data={
            "Pais": True,
            "Tasa por 100k (Incident_Rate)": ":.2f",
            "CFR": ":.2f",
            "Growth_7d": ":.2f",
            "PC1": ":.2f",
            "PC2": ":.2f",
            "Cluster": True
        },
        title="PCA de métricas de COVID-19 por país"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Varianza explicada: PC1 = {pca.explained_variance_ratio_[0]:.2%}, "
               f"PC2 = {pca.explained_variance_ratio_[1]:.2%}")
    with st.expander("Ver *loadings* de PCA (contribución de cada variable)"):
        st.dataframe(loadings.style.format("{:.3f}"))
    # Guardar gráfico a PNG y SVG usando kaleido
    
    png_bytes = fig.to_image(format="png", engine="kaleido")
    svg_bytes = fig.to_image(format="svg", engine="kaleido")

    # Botones de descarga
    st.download_button(
        label="⬇️ Descargar gráfico PCA (PNG)",
        data=png_bytes,
        file_name="pca_grafico.png",
        mime="image/png"
    )

    st.download_button(
        label="⬇️ Descargar gráfico PCA (SVG)",
        data=svg_bytes,
        file_name="pca_grafico.svg",
        mime="image/svg+xml"
    )
    # ----- Descargar resultados PCA en CSV
    csv_pca = df_pca.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar resultados PCA (CSV)",
        data=csv_pca,
        file_name="pca_resultados.csv",
        mime="text/csv"
    )
    # ==================== 4.3
    st.markdown("#### 4.3 Perfiles e interpretación de clústeres")
    profile, interpretations = _interpretar_43(df_pca)
    st.dataframe(profile.style.format("{:.2f}"))
    st.markdown("\n".join(interpretations))
    # Botón para descargar perfiles como CSV
    csv_profile = profile.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar perfiles por cluster (CSV)",
        data=csv_profile,
        file_name="perfiles_clusters.csv",
        mime="text/csv"
    )

    # Botón para descargar interpretaciones como CSV
    df_interpret = pd.DataFrame({
        "Cluster": range(len(interpretations)),
        "Interpretacion": interpretations
    })
    csv_interpret = df_interpret.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar interpretaciones (CSV)",
        data=csv_interpret,
        file_name="interpretaciones_clusters.csv",
        mime="text/csv"
    )

    # Devuelve objetos útiles por si se requiere en tests o descargas
    return df_pca, profile, loadings
