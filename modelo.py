# modelo.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from utils import preprocess, cluster_concepts, monthly_pivot, build_supervised_dataset, rolling_stats

def entrenar_y_predecir(df_raw: pd.DataFrame):
    df = preprocess(df_raw)

    # Clustering automático (oculto al usuario)
    df, vectorizer, kmeans = cluster_concepts(df, n_clusters=8)

    # Pivot mensual
    pv = monthly_pivot(df, use_names=True)

    # Dataset supervisado
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
                f"Estás gastando más en **{cat}** (+${row['delta']:.2f} vs promedio). "
                f"Reducir un 10% en esa categoría ahorraría ~${row['ultimo_mes']*0.10:.2f}."
            )
    if not tips:
        tips.append("Tus gastos del último mes están en línea con tu promedio histórico. ¡Bien ahí!")
    return tips