import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import load_excel
from modelo import (
    entrenar_y_predecir,
    detectar_anomalias,
    sugerencias_ahorro,
    sugerencias_avanzadas
)

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartBudget – Asistente Financiero Inteligente",
    page_icon="💸",
    layout="wide"
)

st.title("💸 SmartBudget – Asistente Financiero Inteligente")
st.caption(
    "Analiza tus gastos diarios con Pandas, visualiza estadísticas con Matplotlib "
    "y predice tus gastos futuros usando aprendizaje automático (Random Forest)."
)

# ------------------------------------------------------------
# CARGA DE ARCHIVO
# ------------------------------------------------------------
st.sidebar.header("📂 Cargar archivo de gastos")
archivo = st.sidebar.file_uploader(
    "Subí tu archivo Excel (.xlsx) con columnas: fecha, concepto, monto, descripcion",
    type=["xlsx"]
)

# Botón descargable para plantilla base
with open("gastos.xlsx", "rb") as template_file:
    st.sidebar.download_button(
        label="📥 Descargar plantilla base",
        data=template_file.read(),
        file_name="gastos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

if archivo is None:
    st.warning("⚠️ No subiste ningún archivo. Por favor, cargá un Excel con tus gastos para continuar.")
    st.stop()

# ------------------------------------------------------------
# LECTURA DEL EXCEL Y CONVERSIÓN DE FECHAS
# ------------------------------------------------------------
df_raw = load_excel(archivo)

# Convertir fechas ANTES de filtrar
df_raw["fecha"] = pd.to_datetime(df_raw["fecha"], errors="coerce")
df_raw = df_raw.dropna(subset=["fecha"])

# ------------------------------------------------------------
# CREAR RANGO AUTOMÁTICO DE FECHAS
# ------------------------------------------------------------
fecha_min = df_raw["fecha"].min().date()
fecha_max = df_raw["fecha"].max().date()

st.sidebar.divider()
st.sidebar.subheader("📆 Filtro de fechas")

rango_fechas = st.sidebar.date_input(
    "Seleccioná un rango",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max,
    key="rango"
)

# Aplicar filtro de fechas si existen 2 fechas válidas
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    inicio, fin = rango_fechas
    inicio = pd.to_datetime(inicio)
    fin = pd.to_datetime(fin)
    df_raw = df_raw[(df_raw["fecha"] >= inicio) & (df_raw["fecha"] <= fin)]

# Vista previa
with st.expander("👀 Vista previa de datos filtrados", expanded=False):
    st.dataframe(df_raw.head(30), use_container_width=True)

# ------------------------------------------------------------
# TIPO DE AGRUPAMIENTO
# ------------------------------------------------------------
st.sidebar.divider()
agrupamiento = st.sidebar.selectbox(
    "📆 Ver estadísticas por:",
    ["Mensual", "Semanal", "Diario"],
    index=0
)

# ------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------
with st.spinner("Entrenando modelo y procesando datos..."):
    try:
        out = entrenar_y_predecir(df_raw)
    except Exception as e:
        st.error(f"❌ Error al preparar/entrenar: {e}")
        st.stop()

df = out["df_limpio"]
pv = out["pivot_mensual"]
pred_mes = out["pred_siguiente_mes"]

# ------------------------------------------------------------
# AGRUPAMIENTO SEGÚN OPCIÓN
# ------------------------------------------------------------
if agrupamiento == "Diario":
    agrupado = df.groupby("fecha")["monto"].sum().reset_index()
    x_col = "fecha"; y_col = "monto"

elif agrupamiento == "Semanal":
    df["semana"] = df["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    agrupado = df.groupby("semana")["monto"].sum().reset_index().rename(columns={"semana":"fecha"})
    x_col = "fecha"; y_col = "monto"

else: # Mensual
    agrupado = (
        df.groupby(df["fecha"].dt.to_period("M").astype(str))["monto"]
        .sum()
        .reset_index()
        .rename(columns={"fecha":"mes","monto":"total"})
    )
    agrupado["fecha"] = agrupado["mes"]
    x_col = "fecha"; y_col = "total"

# ------------------------------------------------------------
# GRÁFICO 1 — EVOLUCIÓN TEMPORAL
# ------------------------------------------------------------
st.markdown("## 📈 Evolución del gasto")

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(agrupado[x_col], agrupado[y_col], marker="o", linewidth=2.5, color="#1f77b4")
ax.set_xlabel(agrupamiento)
ax.set_ylabel("Monto total ($)")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# GRÁFICO 2 — TOP CATEGORÍAS
# ------------------------------------------------------------
st.markdown("## 🏆 Categorías donde más gastaste en el período analizado")

top_cats = (
    df.groupby("categoria_nombre")["monto"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
)

fig_top, ax_top = plt.subplots(figsize=(10,5))
ax_top.barh(top_cats.index, top_cats.values, color="#8E44AD")
ax_top.invert_yaxis()
ax_top.set_xlabel("Monto total gastado ($)")
st.pyplot(fig_top, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# GRÁFICO 3 — COMPARACIÓN ENTRE MESES
# ------------------------------------------------------------
st.markdown("## 🔄 Comparación del último mes vs mes anterior")

if pv.shape[0] >= 2:
    last_two = pv.tail(2)

    fig_cm, ax_cm = plt.subplots(figsize=(10,5))
    index = last_two.columns[:-1]

    ax_cm.bar(index, last_two.iloc[-2][:-1], alpha=0.6,
              label=f"Mes anterior ({last_two.index[-2]})", color="#3498DB")
    ax_cm.bar(index, last_two.iloc[-1][:-1], alpha=0.8,
              label=f"Último mes ({last_two.index[-1]})", color="#E74C3C")

    plt.xticks(rotation=45)
    ax_cm.set_ylabel("Monto ($)")
    ax_cm.legend()
    st.pyplot(fig_cm, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 meses para comparar.")

st.divider()

# ------------------------------------------------------------
# GRÁFICO 4 — DISTRIBUCIÓN DEL ÚLTIMO MES
# ------------------------------------------------------------
st.markdown("### 🍩 Distribución por tipo de gasto (último mes)")

if pv.shape[0] >= 1:
    last_row = pv.drop(columns=["total"], errors="ignore").tail(1).T
    last_row.columns = ["monto"]

    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.pie(last_row["monto"], labels=last_row.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"Distribución {pv.index[-1]}")
    st.pyplot(fig2, use_container_width=True)
else:
    st.write("Sin suficientes datos mensuales.")

st.divider()

# ------------------------------------------------------------
# SECCIÓN DE IA (PREDICCIÓN)
# ------------------------------------------------------------
st.markdown("""
<div style="
    background-color: #f7f7f7;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    margin-top: 10px;
">
<h2 style="text-align:center;color:#000">🔮 Predicción Inteligente del Próximo Mes</h2>
<p style="text-align:center; color:#555;">
Análisis realizado con un modelo Random Forest entrenado sobre tu historial mensual.
</p>
</div>
""", unsafe_allow_html=True)

col_pred1, col_pred2 = st.columns([1, 1])

with col_pred1:
    st.markdown("### 📌 Resultado Principal")
    st.metric("🧾 Gasto estimado próximo mes", f"${pred_mes:,.2f}")

    st.markdown("### 📉 Variación respecto al mes anterior")

    if pv.shape[0] >= 2:
        mes_anterior = pv["total"].iloc[-2]
        dif = pred_mes - mes_anterior
        porcentaje = (dif / mes_anterior) * 100

        flecha = "🔼" if dif > 0 else "🔽"
        color = "red" if dif > 0 else "green"

        st.markdown(
            f"<div style='font-size:20px;'>{flecha} "
            f"<b style='color:{color};'>{porcentaje:.2f}%</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Se necesita al menos un mes previo para comparar.")

with col_pred2:
    st.markdown("### 🧠 Factores según IA")
    modelo = out["modelo_regresion"]
    importancias = pd.Series(modelo.feature_importances_, index=pv.drop(columns=["total"]).columns)
    top_factors = importancias.sort_values(ascending=False).head(3)

    st.markdown("Los rubros que más influyen en tu gasto futuro son:")
    for cat, val in top_factors.items():
        st.markdown(f"- **{cat}** (peso: {val:.2f})")

st.divider()

# ------------------------------------------------------------
# TABS SECUNDARIOS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Datos detallados",
    "📦 Resumen por categoría",
    "⚠️ Anomalías",
    "💡 Sugerencias Inteligentes"
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
    anomalos = daily_anom[daily_anom["anomalia"] == True]

    if anomalos.empty:
        st.success("No se detectaron anomalías 🚀")
    else:
        st.warning("Se encontraron gastos inusuales:")

        for _, row in anomalos.iterrows():
            st.error(
                f"📌 Fecha: {row['fecha'].date()}\n"
                f"💵 Monto total del día: ${row['monto']:.2f}"
            )

with tab4:
    st.subheader("💡 Sugerencias de ahorro (promedios vs último mes)")
    for t in sugerencias_ahorro(pv, top_k=3):
        st.markdown(f"- {t}")

    st.subheader("🧠 Sugerencias avanzadas basadas en datos reales")
    for t in sugerencias_avanzadas(pv):
        st.markdown(f"- {t}")

st.divider()

# ------------------------------------------------------------
# EXPORTACIÓN
# ------------------------------------------------------------
st.subheader("⬇️ Exportar datos procesados")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    csv_raw = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar transacciones limpias (CSV)",
        data=csv_raw,
        file_name="gastos_limpios.csv"
    )

with col_exp2:
    csv_pv = pv.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Descargar resumen mensual (CSV)",
        data=csv_pv,
        file_name="resumen_mensual.csv"
    )