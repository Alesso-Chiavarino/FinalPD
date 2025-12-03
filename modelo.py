import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from utils import preprocess, cluster_concepts, monthly_pivot, build_supervised_dataset, rolling_stats

def entrenar_y_predecir(df_raw: pd.DataFrame):
    df = preprocess(df_raw)
    df, _, _ = cluster_concepts(df, n_clusters=8)
    pv = monthly_pivot(df, use_names=True)

    X, y = build_supervised_dataset(pv)
    if len(X) < 3:
        raise ValueError("Se necesitan al menos 3 meses de datos para entrenar una predicción confiable.")

    modelo = RandomForestRegressor(n_estimators=300, random_state=42)
    modelo.fit(X, y)

    X_pred = pv.drop(columns=["total"], errors="ignore").tail(1)
    pred_siguiente_mes = float(modelo.predict(X_pred)[0])

    by_day, by_cat = rolling_stats(df)

    return {
        "df_limpio": df,
        "pivot_mensual": pv,
        "modelo_regresion": modelo,
        "pred_siguiente_mes": pred_siguiente_mes,
        "gasto_diario": by_day,
        "gasto_por_categoria": by_cat
    }


def detectar_anomalias(df_limpio: pd.DataFrame, contamination: float = 0.05):
    daily = df_limpio.groupby("fecha")["monto"].sum().reset_index()
    X = daily[["monto"]].values
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    daily["anomalia"] = (labels == -1)
    return daily.sort_values("fecha"), iso


def sugerencias_ahorro(pivot_mensual: pd.DataFrame, top_k: int = 3) -> list[str]:
    if pivot_mensual.shape[0] < 2:
        return ["Cargá más meses para generar sugerencias."]

    pv = pivot_mensual.drop(columns=["total"], errors="ignore")
    ultimo = pv.tail(1).T.rename(columns=lambda _: "ultimo_mes")
    prom = pv.iloc[:-1].mean().to_frame(name="prom_hist")

    comp = ultimo.join(prom, how="left").fillna(0)
    comp["delta"] = comp["ultimo_mes"] - comp["prom_hist"]
    comp = comp.sort_values("delta", ascending=False)

    tips = []
    for cat, row in comp.head(top_k).iterrows():
        if row["delta"] > 0:
            tips.append(
                f"• Estás gastando más en **{cat}**.\n"
                f"  - Diferencia: **+${row['delta']:.2f}** respecto a tu promedio.\n"
                f"  - Reducir un 10% en esta categoría te ahorraría **${row['ultimo_mes'] * 0.10:.2f}**."
            )

    if not tips:
        tips.append("Tus gastos del último mes están alineados con tu promedio histórico. ¡Buen trabajo!")

    return tips


def sugerencias_avanzadas(pivot_mensual: pd.DataFrame) -> list[str]:
    tips = []

    if pivot_mensual.shape[0] < 2:
        return ["Cargá más meses para generar sugerencias avanzadas."]

    df = pivot_mensual.drop(columns=["total"], errors="ignore")

    ultimo = df.tail(1).T.rename(columns={df.index[-1]: "ultimo"})
    promedio = df.iloc[:-1].mean(axis=0).to_frame(name="promedio")

    comp = ultimo.join(promedio, how="left")
    comp["delta"] = comp["ultimo"] - comp["promedio"]
    comp["pct"] = comp["ultimo"] / comp["promedio"].replace(0, 1)

    total_ultimo = ultimo["ultimo"].sum()

    for categoria, row in comp.iterrows():
        porcentaje = (row["ultimo"] / total_ultimo) * 100

        if porcentaje > 15:
            tips.append(
                f"• La categoría **{categoria}** representa el **{porcentaje:.1f}%** "
                f"de tu gasto total del mes. Revisá si podés optimizar ese gasto."
            )

        if row["promedio"] > 0 and row["pct"] > 1.25:
            aumento = (row["pct"] - 1) * 100
            tips.append(
                f"• El gasto en **{categoria}** aumentó **{aumento:.1f}%** "
                f"respecto a tu historial.\n"
                f"  ¿Hubo algún gasto extraordinario este mes?"
            )

    if not tips:
        tips.append("Tu comportamiento de gasto es estable y saludable este mes. ¡Excelente trabajo!")

    return tips
