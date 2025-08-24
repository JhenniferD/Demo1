import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# configuración básica
st.set_page_config(page_title="COVID-19 JHU – Métricas y Análisis")
st.title("COVID-19 (JHU)")
st.caption("Fuente: Johns Hopkins CSSE – Daily Report 2022-04-18")


url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"


df = pd.read_csv(url)
 
st.write("DataFrame:")
st.dataframe(df)


# =========================
#2.1. Calcular métricas clave por país: Confirmados, Fallecidos, CFR 
#(muertes/confirmados) y tasas por 100k.
# =========================


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
st.subheader("📈 2.1 Métricas clave por país")
st.dataframe(grouped)


# =========================
# 2.2 Intervalos de confianza para el CFR
# =========================
st.subheader("🧪 Intervalos de confianza para el CFR")


# Controles
colA, colB = st.columns(2)
with colA:
    min_confirm = st.number_input("Mínimo de confirmados por país", min_value=0, value=100, step=50)
with colB:
    conf_level = st.slider("Nivel de confianza", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
alpha = 1 - conf_level


# Filtro de estabilidad
mask_valid = grouped["Confirmados"] >= min_confirm


# Cálculo de IC (intenta Wilson; si no, aproximación normal)
low_ci = np.full(len(grouped), np.nan)
up_ci  = np.full(len(grouped), np.nan)


try:
    from statsmodels.stats.proportion import proportion_confint
    li, ls = proportion_confint(
        grouped.loc[mask_valid, "Fallecidos"].astype(int).values,
        grouped.loc[mask_valid, "Confirmados"].astype(int).values,
        alpha=alpha, method="wilson"
    )
    low_ci[mask_valid.values] = li * 100
    up_ci[mask_valid.values]  = ls * 100
except Exception:
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    p = (grouped.loc[mask_valid, "Fallecidos"] / grouped.loc[mask_valid, "Confirmados"]).astype(float)
    n = grouped.loc[mask_valid, "Confirmados"].astype(float)
    se = np.sqrt(p*(1-p)/n)
    low_ci[mask_valid.values] = (p - z*se) * 100
    up_ci[mask_valid.values]  = (p + z*se) * 100

#

# Añadir columnas sin alterar tus nombres previos
grouped["CFR_LI (%)"] = low_ci
grouped["CFR_LS (%)"] = up_ci


st.caption("Nota: se usa Wilson por defecto; si no está disponible, se aplica aproximación normal.")
st.dataframe(
    grouped.sort_values("CFR (%)", ascending=False)[
        ["Pais", "Confirmados", "Fallecidos", "CFR (%)", "CFR_LI (%)", "CFR_LS (%)", "Tasa casos por 100k (Incident_Rate)"]
    ]
)






# =========================
# 2.3 Test de hipótesis: comparación de CFR entre dos países
# =========================
st.subheader("⚖️ Test de hipótesis: comparación de CFR entre dos países")


# Selectores
pais_options = grouped["Pais"].dropna().sort_values().tolist()
c1, c2, c3 = st.columns([1, 1, 1.2])
with c1:
    pais_a = st.selectbox("País A", pais_options, index=0, key="pais_a")
with c2:
    default_b = 1 if len(pais_options) > 1 else 0
    pais_b = st.selectbox("País B", pais_options, index=default_b, key="pais_b")
with c3:
    alpha_test = st.slider("α (significancia)", 0.001, 0.20, 0.05, 0.005)


if pais_a == pais_b:
    st.info("Selecciona dos países distintos para comparar.")
else:
    fila_a = grouped[grouped["Pais"] == pais_a].iloc[0]
    fila_b = grouped[grouped["Pais"] == pais_b].iloc[0]


    x = np.array([int(fila_a["Fallecidos"]), int(fila_b["Fallecidos"])])  # éxitos
    n = np.array([int(fila_a["Confirmados"]), int(fila_b["Confirmados"])]) # ensayos


    if (n <= 0).any():
        st.warning("Alguno de los países tiene 0 confirmados. No se puede realizar el test.")
    else:
        # z-test de dos proporciones (bilateral)
        try:
            from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
            stat, pval = proportions_ztest(count=x, nobs=n, alternative="two-sided")
            eff = proportion_effectsize(x[0]/n[0], x[1]/n[1])  # h de Cohen
        except Exception:
            from scipy.stats import norm
            p_pool = x.sum() / n.sum()
            se = np.sqrt(p_pool*(1-p_pool)*(1/n[0] + 1/n[1]))
            stat = np.nan if se == 0 else (x[0]/n[0] - x[1]/n[1]) / se
            pval = np.nan if np.isnan(stat) else 2 * (1 - norm.cdf(abs(stat)))
            eff = np.nan


        cfr_a = (x[0] / n[0]) * 100
        cfr_b = (x[1] / n[1]) * 100


        st.markdown(f"""
**Resultados**
- CFR {pais_a}: **{cfr_a:.2f}%**  (Fallecidos: {x[0]} / Confirmados: {n[0]})
- CFR {pais_b}: **{cfr_b:.2f}%**  (Fallecidos: {x[1]} / Confirmados: {n[1]})
- Estadístico z: **{stat:.3f}**
- p-valor: **{pval:.4f}**
- α: **{alpha_test:.3f}**
""")


        if pd.notna(pval) and pval < alpha_test:
            st.success("Conclusión: **Se rechaza H₀**. Hay evidencia de diferencia en los CFR entre los dos países.")
        elif pd.notna(pval):
            st.info("Conclusión: **No se rechaza H₀**. No hay evidencia suficiente de diferencia en los CFR.")
        else:
            st.warning("No fue posible calcular el p-valor por condiciones numéricas.")


        if not np.isnan(eff):
            st.caption(f"Tamaño de efecto (h de Cohen): {eff:.3f} (≈0.2 pequeño, 0.5 mediano, 0.8 grande)")






# =========================
# 2.4 Detección de outliers
# =========================
st.subheader("🔍 2.4 Detección de outliers en CFR (%)")


method = st.radio("Método de detección", ["Z-score", "IQR"], horizontal=True)


outliers = pd.DataFrame()
if method == "Z-score":
    mean_cfr = grouped["CFR (%)"].mean()
    std_cfr = grouped["CFR (%)"].std()
    grouped["Zscore"] = (grouped["CFR (%)"] - mean_cfr) / std_cfr
    outliers = grouped[np.abs(grouped["Zscore"]) > 3]
    st.caption("Se marcan como outliers los países con |Z| > 3.")
    st.dataframe(outliers[["Pais", "CFR (%)", "Zscore"]].sort_values("Zscore"))
else:  # IQR
    q1 = grouped["CFR (%)"].quantile(0.25)
    q3 = grouped["CFR (%)"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = grouped[(grouped["CFR (%)"] < lower) | (grouped["CFR (%)"] > upper)]
    st.caption("Se marcan como outliers los países fuera de [Q1-1.5·IQR, Q3+1.5·IQR].")
    st.dataframe(outliers[["Pais", "CFR (%)"]].sort_values("CFR (%)", ascending=False))


# =========================
# 2.5 Gráfico de control (3σ) de muertes diarias
# =========================
st.subheader("📊 2.5 Gráfico de control (3σ) de muertes diarias")


# Agrupamos muertes por fecha (columna 'Last_Update' o 'Report_Date')
if "Last_Update" in df.columns:
    df["Fecha"] = pd.to_datetime(df["Last_Update"]).dt.date
elif "Report_Date" in df.columns:
    df["Fecha"] = pd.to_datetime(df["Report_Date"]).dt.date
else:
    df["Fecha"] = pd.to_datetime("2022-04-18")  # fallback, ya que es un único archivo


daily_deaths = df.groupby("Fecha")["Deaths"].sum().reset_index()


if daily_deaths.shape[0] > 1:
    mean_d = daily_deaths["Deaths"].mean()
    std_d = daily_deaths["Deaths"].std()
    ucl = mean_d + 3 * std_d
    lcl = max(mean_d - 3 * std_d, 0)  # límite inferior no negativo


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_deaths["Fecha"], daily_deaths["Deaths"], marker="o", label="Muertes diarias")
    ax.axhline(mean_d, color="green", linestyle="--", label="Media")
    ax.axhline(ucl, color="red", linestyle="--", label="UCL (+3σ)")
    ax.axhline(lcl, color="red", linestyle="--", label="LCL (-3σ)")
    ax.set_title("Control de muertes diarias (3σ)")
    ax.set_ylabel("Muertes")
    ax.legend()
    st.pyplot(fig)


    # Mostrar anomalías
    anomalies = daily_deaths[(daily_deaths["Deaths"] > ucl) | (daily_deaths["Deaths"] < lcl)]
    if not anomalies.empty:
        st.warning("⚠️ Se detectaron anomalías fuera de los límites de control:")
        st.dataframe(anomalies)
    else:
        st.success("No se detectaron anomalías en las muertes diarias.")
else:
    st.info("Este archivo corresponde a un solo día, no se puede construir un gráfico de control de serie temporal.")


    # Revisar valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Detectar inconsistencias (ejemplo: valores negativos)
inconsistencias = df[(df["Confirmed"] < 0) | (df["Deaths"] < 0)]
print("\nInconsistencias detectadas:")
print(inconsistencias)

# Gráfico de control: Confirmados por país
grouped = df.groupby("Country_Region", as_index=False).agg({"Confirmed": "sum"})
media = grouped["Confirmed"].mean()
std = grouped["Confirmed"].std()

plt.figure(figsize=(12,6))
plt.plot(grouped["Confirmed"].values, marker="o")
plt.axhline(media, color="green", linestyle="--", label="Media")
plt.axhline(media + 2*std, color="red", linestyle="--", label="Límite superior (2σ)")
plt.axhline(media - 2*std, color="red", linestyle="--", label="Límite inferior (2σ)")
plt.title("Gráfico de Control - Casos Confirmados por País")
plt.legend()
plt.savefig("grafico_control.png", dpi=300)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar dataset desde URL (ejemplo: 18 abril 2022)
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# ==============================
# 5.3 Calidad de datos
# ==============================

# Revisar valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# Detectar inconsistencias (ejemplo: valores negativos)
inconsistencias = df[(df["Confirmed"] < 0) | (df["Deaths"] < 0)]
print("\nInconsistencias detectadas:")
print(inconsistencias)

# Gráfico de control: Confirmados por país
grouped = df.groupby("Country_Region", as_index=False).agg({"Confirmed": "sum"})
media = grouped["Confirmed"].mean()
std = grouped["Confirmed"].std()

plt.figure(figsize=(12,6))
plt.plot(grouped["Confirmed"].values, marker="o")
plt.axhline(media, color="green", linestyle="--", label="Media")
plt.axhline(media + 2*std, color="red", linestyle="--", label="Límite superior (2σ)")
plt.axhline(media - 2*std, color="red", linestyle="--", label="Límite inferior (2σ)")
plt.title("Gráfico de Control - Casos Confirmados por País")
plt.legend()
plt.savefig("grafico_control.png", dpi=300)
plt.show()

# ==============================
# 5.4 Exportación
# ==============================

# Exportar tabla procesada
grouped.to_csv("resumen_confirmados.csv", index=False)

# Guardar gráfico en SVG también
plt.savefig("grafico_control.svg")

# ==============================
# 5.5 Narrativa automática
# ==============================

# País con más confirmados
top_confirmed = grouped.loc[grouped["Confirmed"].idxmax()]

# País con más muertes
grouped_deaths = df.groupby("Country_Region", as_index=False).agg({"Deaths": "sum"})
top_deaths = grouped_deaths.loc[grouped_deaths["Deaths"].idxmax()]

narrativa = (
    f"El análisis muestra que {top_confirmed['Country_Region']} tiene la mayor cantidad "
    f"de casos confirmados ({top_confirmed['Confirmed']:,}), "
    f"mientras que {top_deaths['Country_Region']} presenta el mayor número de muertes "
    f"({top_deaths['Deaths']:,}). Estos hallazgos resaltan diferencias importantes en "
    f"la propagación y el impacto del virus entre países."
)

print("\nNarrativa automática:")
print(narrativa)
