import pandas as pd
import streamlit as st
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import io

# === 3.1 Generación de Series de Tiempo con Suavizado de 7 Días ===
def mostrar_series_tiempo(df):
    # Título y descripción
    st.subheader('🌍 3.1 Generación de Series de Tiempo con Suavizado de 7 Días')
    st.markdown("""
    **Descripción de la actividad:**  
    Este gráfico muestra los **casos confirmados** y **muertes suavizadas** a través de una media móvil de 7 días. El suavizado es importante para eliminar fluctuaciones de datos y observar tendencias más claras.

    Se generan los datos de manera simulada para los días previos a la fecha final y se muestran en un gráfico de líneas para facilitar la visualización de las tendencias.

    """)

    st.caption("⚠️ Solo se dispone de los reportes diarios de: 28/02/2022, 01/03/2022 y 19/04/2022.")

    # Selección de país
    country = st.selectbox("Seleccione un país para ver sus datos suavizados:", df['Country_Region'].unique())

    # Filtrar los datos por el país seleccionado
    df_country = df[df['Country_Region'] == country]

    # Agrupar por fecha y sumar los casos confirmados
    df_country_grouped = df_country.groupby('Last_Update').agg({'Confirmed': 'sum', 'Deaths': 'sum'}).reset_index()

    # Obtener la última fecha (Last_Update) en los datos del país
    last_update = df_country_grouped['Last_Update'].max()
    aux_date = last_update.strftime('%d/%m/%Y')

    colA, colB = st.columns(2)
    with colA:
        st.metric("Confirmados (día base)", df_country_grouped['Confirmed'].iloc[-1])
    with colB:
        st.metric("Muertes (día base)", df_country_grouped['Deaths'].iloc[-1])

    st.write(f"*Fecha base:* **{aux_date}** -- Fuente: reporte diario JHU (1 día).")

    # Generar las 7 fechas: 6 fechas anteriores a la de Last_Update
    date_list = [last_update]  # Agregamos la última fecha (Last_Update)
    for i in range(6):
        date_list.append(last_update - timedelta(days=i+1))  # Restamos un día por cada iteración

    # Convertir las fechas a formato "DD/MM/YYYY"
    date_list = [date.strftime('%d/%m/%Y') for date in reversed(date_list)]  # Reversed para tenerlas en orden ascendente

    # Crear la lista de valores para "Confirmed" en orden ascendente
    confirmed_values = [df_country_grouped['Confirmed'].iloc[-1]]  # Empezamos con el valor más reciente
    for i in range(6):
        # Generar un incremento del 5% a 20% de lo último (en función de los datos reales)
        percent = np.random.uniform(0.05, 0.20)  # Un rango más ajustado
        new_value = max(confirmed_values[-1] * (1 - percent), 0)  # Restamos un porcentaje para los días previos
        confirmed_values.append(new_value)

    # Ordenar los valores de casos confirmados de menor a mayor
    confirmed_values = sorted(confirmed_values)

    # Crear la lista de valores para "Deaths" de manera similar
    deaths_values = [df_country_grouped['Deaths'].iloc[-1]]  # Empezamos con el valor más reciente
    for i in range(6):
        # Generar un incremento del 2% a 10% de lo último (en función de los datos reales)
        percent = np.random.uniform(0.02, 0.10)  # Un rango más ajustado
        new_value = max(deaths_values[-1] * (1 - percent), 0)  # Restamos un porcentaje para los días previos
        deaths_values.append(new_value)

    # Ordenar los valores de muertes de menor a mayor
    deaths_values = sorted(deaths_values)

    # Crear DataFrame con índices que comienzan desde 1
    df_display = pd.DataFrame({
        'Fecha': date_list,
        'Casos Confirmados': np.array(confirmed_values).astype(int),  # Convertir a enteros
        'Muertes': np.array(deaths_values).astype(int)  # Convertir a enteros
    })
    df_display.index += 1  # Cambiar el índice para que empiece desde 1

    # Mostrar la tabla con las fechas y los valores
    st.write("### Tabla de Casos Confirmados y Muertes por Fecha:")
    st.write(df_display)

    # --- Función para formatear el eje Y en millares ---
    def thousands(x, pos):
        return '%1.0fK' % (x * 1e-3)  # Convertir a miles y mostrar 'K'

    # --- construir series con índice de fechas ---
    idx = pd.to_datetime(date_list, dayfirst=True)
    confirmed_sr = pd.Series(confirmed_values, index=idx)
    deaths_sr = pd.Series(deaths_values, index=idx)

    # --- Suavizado 7 días (rolling mean) ---
    confirmed_ma7 = confirmed_sr.rolling(window=7, min_periods=1, center=True).mean()
    deaths_ma7 = deaths_sr.rolling(window=7, min_periods=1, center=True).mean()

    # --- plot lado a lado ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Confirmados
    ax1.plot(idx, confirmed_sr.values, linewidth=1.2, alpha=0.35, label='Serie original')
    ax1.plot(idx, confirmed_ma7.values, linewidth=2.5, label='MA 7 días')
    ax1.set_title(f"Casos Confirmados Suavizados en {country}")
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Número de casos en millares')
    ax1.legend()
    for label in ax1.get_xticklabels(): 
        label.set_rotation(45)

    # Aplicar formateo a los valores del eje Y en 'ax1'
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands))

    # Muertes
    ax2.plot(idx, deaths_sr.values, linewidth=1.2, alpha=0.35, label='Serie original', color='tab:orange')
    ax2.plot(idx, deaths_ma7.values, linewidth=2.5, label='MA 7 días', color='tab:red')
    ax2.set_title(f"Muertes Suavizadas en {country}")
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Número de muertes')
    ax2.legend()
    for label in ax2.get_xticklabels(): 
        label.set_rotation(45)

    # Mostrar los gráficos
    st.pyplot(fig)


    # === EXPORTACIÓN ===
    # CSV
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar datos en CSV",
        data=csv,
        file_name=f"series_tiempo_{country}.csv",
        mime="text/csv"
    )

    # PNG
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png")
    st.download_button(
        label="⬇️ Descargar gráfico en PNG",
        data=buf_png.getvalue(),
        file_name=f"series_tiempo_{country}.png",
        mime="image/png"
    )

    # SVG
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg")
    st.download_button(
        label="⬇️ Descargar gráfico en SVG",
        data=buf_svg.getvalue(),
        file_name=f"series_tiempo_{country}.svg",
        mime="image/svg+xml"
    )

# === 3.2 y 3.3 Modelado ETS y Validación ===
def mostrar_modelado_forecast(url):
    # Funciones de evaluación
    def mae(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true)-np.array(y_pred)))
    def mape(y_true, y_pred):
        denom = np.where(np.array(y_true)==0,1.0,np.array(y_true))
        return np.mean(np.abs((np.array(y_true)-np.array(y_pred))/denom))*100

    # Simulación histórica
    def simulate_history(last_value, days=60, daily_growth=0.03, noise=0.0, last_date=pd.Timestamp("2022-04-18")):
        values_backward = [float(max(last_value,0))]
        rng = np.random.default_rng(42)
        for _ in range(days-1):
            g = daily_growth
            if noise>0: g = max(-0.5, g + rng.normal(0, noise*daily_growth))
            prev = values_backward[-1]/(1+max(g,-0.9))
            values_backward.append(max(prev,0.0))
        values = list(reversed(values_backward))
        for i in range(1,len(values)): values[i] = max(values[i],values[i-1])
        start_date = pd.to_datetime(last_date) - pd.Timedelta(days=days-1)
        return pd.Series(values,index=pd.date_range(start=start_date,periods=days,freq="D"))

    st.subheader("🧪 Modelado y Proyección COVID-19")

    try:
        raw = pd.read_csv(url)
    except FileNotFoundError:
        st.error("No se encontró la url deseada. ¡Revisar bien la dirección url!")
        st.stop()

    # Limpieza básica
    raw["Last_Update"] = pd.to_datetime(raw["Last_Update"], errors="coerce")
    raw = raw.dropna(subset=["Last_Update"])
    if raw.empty:
        st.error("El archivo se cargó pero no tiene fechas válidas en 'Last_Update'.")
        st.stop()

    last_date = raw["Last_Update"].max().normalize()
    daily = (raw.loc[raw["Last_Update"].dt.normalize() == last_date,
                    ["Country_Region", "Confirmed", "Deaths"]]
                .groupby("Country_Region", as_index=False)
                .sum())

    if daily.empty:
        st.error("No se encontraron registros para la fecha del archivo.")
        st.stop()

    # UI: selección de país
    st.subheader("Datos del día base (18/04/2022)")
    country = st.selectbox("Selecciona un país", options=sorted(daily["Country_Region"].unique()))

    row = daily[daily["Country_Region"]==country].iloc[0]
    base_confirmed, base_deaths = int(row["Confirmed"]), int(row["Deaths"])

    colA, colB = st.columns(2)
    with colA: st.metric("Confirmados", f"{base_confirmed:,}")
    with colB: st.metric("Muertes", f"{base_deaths:,}")

    st.subheader("Parámetros para generar histórico simulado")
    sim_days = st.slider("Días histórico", 30, 120, 60)
    growth_confirmed = st.slider("Crecimiento Confirmados", 0.0, 0.08, 0.03, 0.005)
    growth_deaths = st.slider("Crecimiento Muertes", 0.0, 0.08, 0.02, 0.005)
    noise_level = st.slider("Ruido", 0.0, 0.5, 0.1, 0.01)

    sim_confirmed = simulate_history(base_confirmed,days=sim_days,daily_growth=growth_confirmed,noise=noise_level,last_date=last_date)
    sim_deaths = simulate_history(base_deaths,days=sim_days,daily_growth=growth_deaths,noise=noise_level,last_date=last_date)
    hist_df = pd.DataFrame({"Confirmed":sim_confirmed,"Deaths":sim_deaths})

    st.subheader("Histórico SIMULADO")
    st.line_chart(hist_df)
        # Botón para exportar histórico simulado
    st.download_button("📥 Descargar histórico simulado (CSV)",
                       data=hist_df.to_csv(index=True).encode("utf-8"),
                       file_name="historico_simulado.csv",
                       mime="text/csv")

    # --- 3.2 Modelado ETS ---
    st.subheader("3.2 Implementación del Modelo ETS para pronóstico")
    st.markdown("Se aplica el modelo ETS para proyectar **casos** o **muertes** a 14 días.")
    target = st.selectbox("Variable a pronosticar", ["Confirmed","Deaths"])
    series = hist_df[target]
    if len(series)<30: st.error("Se requieren al menos 30 días simulados para entrenar ETS."); st.stop()

    h=14
    train, test = series.iloc[:-h], series.iloc[-h:]
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal=None)
        fit = model.fit()
        preds = fit.forecast(h)
    except Exception as e: st.error(f"Error al ajustar ETS: {e}"); st.stop()

    # --- 3.3 Validación ---
    st.subheader("3.3 Validación del Modelo con Backtesting (MAE / MAPE)")
    st.markdown("Se comparan las predicciones con los valores reales del test set para evaluar precisión.")
    mae_val, mape_val = mae(test.values, preds.values), mape(test.values, preds.values)
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae_val:,.2f}")
    c2.metric("MAPE", f"{mape_val:.2f}%")

    fig1, ax1 = plt.subplots(figsize=(9,4.5))
    ax1.plot(train.index, train.values,label="Entrenamiento")
    ax1.plot(test.index, test.values,label="Real (test)")
    ax1.plot(preds.index, preds.values,"--",label="Pronóstico (test)")
    ax1.set_title(f"Backtesting ETS - {target}")
    ax1.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    # Exportar gráfico de backtesting en PNG y SVG
    buf_png, buf_svg = io.BytesIO(), io.BytesIO()
    fig1.savefig(buf_png, format="png", bbox_inches="tight")
    fig1.savefig(buf_svg, format="svg", bbox_inches="tight")
    st.download_button("📸 Descargar gráfico backtesting (PNG)", data=buf_png.getvalue(), file_name="backtesting.png", mime="image/png")
    st.download_button("📸 Descargar gráfico backtesting (SVG)", data=buf_svg.getvalue(), file_name="backtesting.svg", mime="image/svg+xml")

    # --- Proyección a 14 días ---
    st.subheader("Proyección a 14 días hacia adelante")
    fit_full = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    future_fc = fit_full.forecast(h)
    future_df = pd.DataFrame({f"Forecast_{target}":future_fc})
    st.line_chart(future_df)
    st.dataframe(future_df.style.format("{:,.0f}"))
    st.info("⚠️ Este análisis es ilustrativo, basado en un histórico simulado debido a que solo se cuenta con un día real.")

    # Botón para exportar proyección
    st.download_button("📥 Descargar proyección (CSV)",
                       data=future_df.to_csv(index=True).encode("utf-8"),
                       file_name="proyeccion.csv",
                       mime="text/csv")

    # Exportar gráfico de proyección en PNG y SVG
    fig2, ax2 = plt.subplots(figsize=(9,4.5))
    ax2.plot(series.index, series.values, label="Histórico")
    ax2.plot(future_df.index, future_df.values, "--", label="Pronóstico 14d")
    ax2.set_title(f"Proyección ETS - {target}")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    buf_png2, buf_svg2 = io.BytesIO(), io.BytesIO()
    fig2.savefig(buf_png2, format="png", bbox_inches="tight")
    fig2.savefig(buf_svg2, format="svg", bbox_inches="tight")
    st.download_button("📸 Descargar gráfico proyección (PNG)", data=buf_png2.getvalue(), file_name="proyeccion.png", mime="image/png")
    st.download_button("📸 Descargar gráfico proyección (SVG)", data=buf_svg2.getvalue(), file_name="proyeccion.svg", mime="image/svg+xml")

# ==================================
# === 3.4 Mostrar bandas de confianza en la gráfica de forecast ===
# ==================================
def bandas_confianza(df):
   # Definir el horizonte de predicción (1 a 30 días)
    h = st.slider("Elegir el horizonte de predicción (en días):", 1, 30, 14)

    # --- Proyección a h días --- 
    st.subheader(f"Proyección a {h} días hacia adelante")

    # Selección de la variable a pronosticar con un key único
    target = st.selectbox("Variable a pronosticar", ["Confirmed", "Deaths"], key="target_selectbox")

    # Extraemos solo los últimos 30 días de datos para la variable seleccionada
    series = df[target].tail(30)  # Solo toma los últimos 30 días de la serie

    # Ajuste completo del modelo ETS con la serie seleccionada
    fit_full = ExponentialSmoothing(series, trend='add', seasonal=None).fit()

    # Realiza la proyección
    future_fc = fit_full.forecast(h)

    forecast_error = np.std(fit_full.resid)

    # Cambiar el intervalo de confianza al 99% (z-score para 99% es 2.576)
    z_score = 2.576
    upper_bound = future_fc + z_score * forecast_error
    lower_bound = future_fc - z_score * forecast_error

    # **Asegurarse de que no haya valores negativos solo en ciertas condiciones (como "Confirmed")**
    if target == "Confirmed":
        future_fc = np.maximum(future_fc, 0)  # Forzar cero solo si es negativo
        upper_bound = np.maximum(upper_bound, 0)
        lower_bound = np.maximum(lower_bound, 0)

    # Crea la proyección futura con las bandas de confianza
    future_df = pd.DataFrame({
        f"Forecast_{target}": future_fc,
        f"Upper_{target}": upper_bound,
        f"Lower_{target}": lower_bound
    })

    # **Generar fechas consecutivas a partir de la última fecha en df["Last_Update"]**
    last_date = pd.to_datetime(df["Last_Update"][0])  # Obtener la última fecha real del DataFrame
    future_dates = [last_date + timedelta(days=i+1) for i in range(h)]  # Generar fechas consecutivas

    # Asignar las fechas al índice del DataFrame
    future_df.index = future_dates

    # Graficar la proyección con las bandas de confianza
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))

    # Graficar el pronóstico
    ax2.plot(future_df.index, future_df[f"Forecast_{target}"], label="Pronóstico", color='blue')

    # Graficar la banda de confianza
    ax2.fill_between(future_df.index, future_df[f"Lower_{target}"], future_df[f"Upper_{target}"], color='orange', alpha=0.3, label="Banda de Confianza 99%")

    ax2.set_title(f"Proyección a {h} días con Bandas de Confianza - {target}")
    
    # Mejorar la visualización de los ejes
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Valor de Confirmados (miles)')
    
    # Formato de notación en miles en el eje Y
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}K".format(int(x / 1000))))
    
    ax2.tick_params(axis='x', rotation=45)  # Rotar las etiquetas del eje X
    ax2.tick_params(axis='y', labelsize=10)  # Ajustar el tamaño de las etiquetas del eje Y
    
    ax2.legend(loc='upper left')
    st.pyplot(fig2)

    # Mostrar los datos proyectados con formato
    st.dataframe(future_df.style.format("{:,.0f}"))
    return

