import streamlit as st
import altair as alt
import plotly.express as px

def mostrar_topn_mapa(df):
    st.subheader("Top-N países por métricas COVID")

    # --- Diccionario de métricas ---
    metricas = {
        "Confirmados": "Confirmed",
        "Muertes": "Deaths",
        "Tasa de fatalidad": "Case_Fatality_Ratio"
    }

    # --- Controles ---
    opcion = st.selectbox("Selecciona la métrica", list(metricas.keys()))
    variable = metricas[opcion]

    top_n = st.slider("Número de países (Top-N)", 3, 20, 10)

    # --- Agrupar por país ---
    if variable in ["Confirmed", "Deaths"]:
        df_grouped = df.groupby("Country_Region")[variable].sum().reset_index()
    else:  # Case_Fatality_Ratio
        df_grouped = df.groupby("Country_Region")[variable].mean().reset_index()

    # --- Ordenar y filtrar top ---
    df_top = df_grouped.sort_values(by=variable, ascending=False).head(top_n)

    # --- Gráfico de barras ---
    chart = alt.Chart(df_top).mark_bar().encode(
        x=alt.X(variable, title=opcion),
        y=alt.Y("Country_Region", sort="-x", title="País"),
        tooltip=["Country_Region", variable]
    ).properties(
        title=f"Top {top_n} países por {opcion}"
    )

    st.altair_chart(chart, use_container_width=True)

    # --- 🌍 Mapa interactivo ---
    st.subheader(f"Mapa mundial de {opcion}")

    fig = px.choropleth(
        df_grouped,
        locations="Country_Region",
        locationmode="country names",
        color=variable,
        hover_name="Country_Region",
        color_continuous_scale="Reds",
        title=f"{opcion} por país",
        labels={variable: opcion}
    )

    st.plotly_chart(fig, use_container_width=True)
