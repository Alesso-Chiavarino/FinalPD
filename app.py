# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import load_excel
from modelo import entrenar_y_predecir, detectar_anomalias, sugerencias_ahorro

# ---- CONFIGURACIÓN ----
st.set_page_config(
    page_title="SmartBudget – Asistente Financiero Inteligente",
    page_icon="💸",
    layout="wide"
)

st.title("💸 SmartBudget – Asistente Financiero Inteligente")
st.caption("Analiza tus gastos diarios con **Pandas**, visualiza estadísticas con **Matplotlib** y predice tus gastos futuros usando **aprendizaje automático (Random Forest)**.")

# ---- SIDEBAR: CARGA DE ARCHIVO ----
st.sidebar.header("📂 Cargar archivo de gastos")
archivo = st.sidebar.file_uploader(
    "Subí tu archivo Excel (.xlsx) con columnas: fecha, concepto, monto, [descripcion]",
    type=["xlsx"]
)

# ---- NUEVO: Tipo de visualización ----
st.sidebar.divider()
agrupamiento = st.sidebar.selectbox(
    "📆 Ver estadísticas por:",
    ["Mensual", "Semanal", "Diario"],
    index=0
)

if archivo is None:
    st.warning("⚠️ No subiste ningún archivo. Por favor, cargá un Excel con tus gastos para continuar.")
    st.stop()

# ---- LECTURA DE DATOS ----
df_raw = load_excel(archivo)

with st.expander("👀 Vista previa de datos cargados", expanded=False):
    st.dataframe(df_raw.head(30), use_container_width=True)

# ---- ENTRENAMIENTO ----
with st.spinner("Entrenando modelo y procesando datos..."):
    try:
        out = entrenar_y_predecir(df_raw)  # Se quita n_clusters visible
    except Exception as e:
        st.error(f"❌ Error al preparar/entrenar: {e}")
        st.stop()

df = out["df_limpio"]
pv = out["pivot_mensual"]
pred_mes = out["pred_siguiente_mes"]

# ---- GRÁFICOS PRINCIPALES ----
col1, col2, col3 = st.columns([2, 2, 1])

# 🔹 Configurar agrupamiento (diario, semanal o mensual)
if agrupamiento == "Diario":
    agrupado = df.groupby("fecha")["monto"].sum().reset_index()
    x_col = "fecha"
    y_col = "monto"
elif agrupamiento == "Semanal":
    df["semana"] = df["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    agrupado = df.groupby("semana")["monto"].sum().reset_index().rename(columns={"semana": "fecha"})
    x_col, y_col = "fecha", "monto"
else:  # Mensual
    agrupado = df.groupby(df["fecha"].dt.to_period("M").astype(str))["monto"].sum().reset_index().rename(columns={"fecha": "mes", "monto": "total"})
    agrupado["fecha"] = agrupado["mes"]
    x_col, y_col = "fecha", "total"

# ---- Gráfico de evolución temporal ----
with col1:
    st.subheader(f" Evolución {agrupamiento.lower()} del gasto total")
    fig, ax = plt.subplots()
    ax.plot(agrupado[x_col], agrupado[y_col], marker="o", color="#2196F3")
    ax.set_xlabel(agrupamiento)
    ax.set_ylabel("Monto total ($)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

# ---- Distribución por categoría ----
with col2:
    st.subheader("Distribución por tipo de gasto (último mes)")
    if pv.shape[0] >= 1:
        last_row = pv.drop(columns=["total"], errors="ignore").tail(1).T
        last_row.columns = ["monto"]
        fig2, ax2 = plt.subplots()
        if last_row["monto"].sum() > 0:
            ax2.pie(last_row["monto"], labels=last_row.index, autopct="%1.1f%%", startangle=90)
        ax2.set_title(f"Distribución {pv.index[-1]}")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.write("Sin suficientes datos mensuales.")

# ---- Predicción ----
with col3:
    st.subheader("Predicción IA")
    st.metric(label=f"Gasto estimado próximo mes", value=f"${pred_mes:,.2f}")

st.divider()

# ---- SECCIONES DETALLE ----
tab1, tab2, tab3 = st.tabs([
    "📊 Datos detallados",
    "📦 Resumen por categoría",
    "⚠️ Anomalías y Sugerencias"
])

with tab1:
    st.subheader(f"Gasto {agrupamiento.lower()} detallado")
    st.dataframe(agrupado, use_container_width=True, height=320)

with tab2:
    st.subheader("Resumen mensual por categorías")
    st.dataframe(pv, use_container_width=True, height=350)

with tab3:
    st.subheader("🚨 Detección de anomalías")
    daily_anom, _iso = detectar_anomalias(df)
    st.dataframe(daily_anom[daily_anom["anomalia"]], use_container_width=True, height=240)
    st.caption("Se marcan días con gastos atípicos usando IsolationForest.")

    st.subheader("💡 Sugerencias de ahorro personalizadas")
    tips = sugerencias_ahorro(pv, top_k=3)
    for t in tips:
        st.markdown(f"- {t}")

st.divider()

# ---- EXPORTACIÓN ----
st.subheader("⬇️ Exportar datos procesados")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv_raw = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar transacciones limpias (CSV)", data=csv_raw, file_name="gastos_limpios.csv")

with col_exp2:
    csv_pv = pv.to_csv(index=True).encode("utf-8")
    st.download_button("Descargar resumen mensual (CSV)", data=csv_pv, file_name="resumen_mensual.csv")