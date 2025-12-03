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
st.caption(
    "Analiza tus gastos diarios con **Pandas**, visualiza estadísticas con **Matplotlib** "
    "y predice tus gastos futuros usando **aprendizaje automático (Random Forest)**."
)

# ---- SIDEBAR: CARGA DE ARCHIVO ----
st.sidebar.header("📂 Cargar archivo de gastos")
archivo = st.sidebar.file_uploader(
    "Subí tu archivo Excel (.xlsx) con columnas: fecha, concepto, monto, [descripcion]",
    type=["xlsx"]
)

# ---- Tipo de visualización ----
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
        out = entrenar_y_predecir(df_raw)
    except Exception as e:
        st.error(f"❌ Error al preparar/entrenar: {e}")
        st.stop()

df = out["df_limpio"]
pv = out["pivot_mensual"]
pred_mes = out["pred_siguiente_mes"]

# ============================================================================ #
#                           SECCIÓN PRINCIPAL VISUAL                           #
# ============================================================================ #

# ------ AGRUPAMIENTO PARA GRAFICO PRINCIPAL ------
if agrupamiento == "Diario":
    agrupado = df.groupby("fecha")["monto"].sum().reset_index()
    x_col = "fecha"; y_col = "monto"

elif agrupamiento == "Semanal":
    df["semana"] = df["fecha"].dt.to_period("W").apply(lambda r: r.start_time)
    agrupado = df.groupby("semana")["monto"].sum().reset_index().rename(columns={"semana": "fecha"})
    x_col, y_col = "fecha", "monto"

else:  # Mensual
    agrupado = (
        df.groupby(df["fecha"].dt.to_period("M").astype(str))["monto"]
        .sum()
        .reset_index()
        .rename(columns={"fecha": "mes", "monto": "total"})
    )
    agrupado["fecha"] = agrupado["mes"]
    x_col, y_col = "fecha", "total"


# --------------------- GRAFICO 1: EVOLUCIÓN ---------------------
st.markdown("## 📈 Evolución del gasto")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(agrupado[x_col], agrupado[y_col], marker="o", linewidth=2.5, color="#1f77b4")
ax.set_xlabel(agrupamiento)
ax.set_ylabel("Monto total ($)")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig, use_container_width=True)

st.divider()


# --------------------- GRAFICO 2: TOP CATEGORÍAS ---------------------
st.markdown("## 🏆 Categorías donde más gastaste en todo el período")

top_cats = (
    df.groupby("categoria_nombre")["monto"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
)

fig_top, ax_top = plt.subplots(figsize=(10, 5))
ax_top.barh(top_cats.index, top_cats.values, color="#8E44AD")
ax_top.invert_yaxis()
ax_top.set_xlabel("Monto total gastado ($)")
ax_top.set_title("Top categorías del período analizado")
st.pyplot(fig_top, use_container_width=True)

st.divider()


# ---------------- GRAFICO 3: COMPARACIÓN MES A MES ----------------
st.markdown("## 🔄 Comparación del último mes vs mes anterior")

if pv.shape[0] >= 2:
    last_two = pv.tail(2)

    fig_cm, ax_cm = plt.subplots(figsize=(10, 5))
    index = last_two.columns[:-1]  # sin la columna total

    ax_cm.bar(index, last_two.iloc[-2][:-1], alpha=0.6, label=f"Mes anterior ({last_two.index[-2]})", color="#3498DB")
    ax_cm.bar(index, last_two.iloc[-1][:-1], alpha=0.8, label=f"Último mes ({last_two.index[-1]})", color="#E74C3C")

    plt.xticks(rotation=45)
    ax_cm.set_ylabel("Monto ($)")
    ax_cm.legend()
    st.pyplot(fig_cm, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 meses para comparar.")

st.divider()


# ---------------- SECCIÓN: DISTRIBUCIÓN --------------------
st.markdown("## 🍩 Distribución por tipo de gasto (último mes)")

colA, _ = st.columns([3, 1])

with colA:
    if pv.shape[0] >= 1:
        last_row = pv.drop(columns=["total"], errors="ignore").tail(1).T
        last_row.columns = ["monto"]

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        if last_row["monto"].sum() > 0:
            ax2.pie(
                last_row["monto"],
                labels=last_row.index,
                autopct="%1.1f%%",
                startangle=90
            )
        ax2.set_title(f"Distribución {pv.index[-1]}")
        st.pyplot(fig2, use_container_width=True)
    else:
        st.write("Sin suficientes datos mensuales.")

st.divider()

# ======================================================================== #
#                          🔮 SECCIÓN IA PREMIUM                            #
# ======================================================================== #

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

# ---- Layout interno ----
col_pred1, col_pred2 = st.columns([1, 1])


# -------------------- LEFT SIDE: METRICS --------------------
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
            f"<div style='font-size:20px;'>"
            f"{flecha} <b style='color:{color};'>{porcentaje:.2f}%</b>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Se necesita al menos un mes previo para comparar.")


# -------------------- RIGHT SIDE: TOP FACTORS --------------------
with col_pred2:
    st.markdown("### 🧠 Factores según IA")

    modelo = out["modelo_regresion"]
    importancias = pd.Series(modelo.feature_importances_, index=pv.drop(columns=["total"]).columns)

    top_factors = importancias.sort_values(ascending=False).head(3)

    st.markdown("Los rubros que más influyen en tu gasto futuro son:")

    for cat, val in top_factors.items():
        st.markdown(f"- **{cat}** (peso: {val:.2f})")

st.divider()

# ============================================================================ #
#                                TABS SECUNDARIOS                              #
# ============================================================================ #

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

# ============================================================================ #
#                                EXPORTACIÓN                                   
# ============================================================================ #

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