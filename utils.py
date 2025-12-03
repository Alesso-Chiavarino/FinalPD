# utils.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ---------- Validación mínima ----------
REQUIRED_COLUMNS = ["fecha", "concepto", "monto"]
OPTIONAL_COLUMNS = ["descripcion"]

# ---------- Limpieza de texto ----------
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s) 
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Lectura de archivo Excel ----------
def load_excel(file_or_path) -> pd.DataFrame:
    df = pd.read_excel(file_or_path)
    return df

# ---------- Preprocesamiento ----------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza los nombres de las columnas
    cols = [c.lower().strip() for c in df.columns]
    df = df.set_axis(cols, axis=1, copy=False)

    # Verifica columnas necesarias
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. "
            f"Debés incluir: {REQUIRED_COLUMNS} y opcional {OPTIONAL_COLUMNS}"
        )

    # Convierte fechas y ordena
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    df = df.sort_values("fecha")

    # Limpieza de texto
    df["concepto"] = df["concepto"].apply(_normalize_text)
    if "descripcion" in df.columns:
        df["descripcion"] = df["descripcion"].apply(_normalize_text)
    else:
        df["descripcion"] = ""

    # Asegura que los montos sean numéricos y positivos
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0.0)
    df = df[df["monto"] != 0]
    df["monto"] = df["monto"].abs()

    # Crea columnas de fecha útiles
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    return df.reset_index(drop=True)

# ---------- Clustering de conceptos ----------
def cluster_concepts(df: pd.DataFrame, n_clusters: int = 8):
    # Combina concepto y descripción
    corpus = (df["concepto"].fillna("") + " " + df["descripcion"].fillna("")).values

    # Stopwords básicas en español
    spanish_stopwords = [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
        "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
        "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
        "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
        "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
        "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
        "qué", "unos", "yo", "otro", "otras", "otra", "él", "ella", "ellos",
        "ellas", "usted", "ustedes", "mi", "tu", "te", "ti", "su"
    ]

    # Vectorización de texto
    vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, min_df=2)
    X = vectorizer.fit_transform(corpus)

    # Ajustar número de clusters según tamaño del dataset
    n_clusters = max(2, min(n_clusters, max(2, X.shape[0] // 20)))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["categoria_auto"] = labels

    # ---------- Nombres amigables para cada categoría ----------
    inv_vocab = {i: t for t, i in vectorizer.vocabulary_.items()}
    try:
        centers = kmeans.cluster_centers_
        top_words = []
        for i in range(centers.shape[0]):
            idx = centers[i].argsort()[-5:][::-1]
            terms = [inv_vocab.get(j, "") for j in idx]
            label_name = ", ".join([t for t in terms if t][:2]) or f"cat_{i}"
            top_words.append(label_name)
        map_names = {i: top_words[i] for i in range(len(top_words))}
        df["categoria_nombre"] = df["categoria_auto"].map(map_names)
    except Exception:
        df["categoria_nombre"] = "categoria"

    return df, vectorizer, kmeans

# ---------- Pivot mensual ----------
def monthly_pivot(df: pd.DataFrame, use_names: bool = True) -> pd.DataFrame:
    col = "categoria_nombre" if use_names and "categoria_nombre" in df.columns else "categoria_auto"
    pv = df.pivot_table(
        index="mes",
        columns=col,
        values="monto",
        aggfunc="sum",
        fill_value=0.0
    ).sort_index()
    pv.index = pd.PeriodIndex(pv.index, freq="M").astype(str)
    pv["total"] = pv.sum(axis=1)
    return pv

# ---------- Dataset supervisado (para predicción) ----------
def build_supervised_dataset(pv: pd.DataFrame):
    # X: categorías sin total. y: total del siguiente mes
    X = pv.drop(columns=["total"], errors="ignore")
    total = pv["total"].copy()
    y = total.shift(-1)
    X = X.iloc[:-1].copy()
    y = y.iloc[:-1].copy()
    return X, y

# ---------- Estadísticas adicionales ----------
def rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Gasto diario total
    by_day = df.groupby("fecha")["monto"].sum().rename("gasto_diario")
    # Gasto acumulado por categoría
    by_cat = df.groupby("categoria_nombre")["monto"].sum().sort_values(ascending=False)
    return by_day.to_frame(), by_cat.to_frame(name="gasto_por_categoria")