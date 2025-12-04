import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

REQUIRED_COLUMNS = ["fecha", "concepto", "monto"]
OPTIONAL_COLUMNS = ["descripcion"]

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def load_excel(file_or_path) -> pd.DataFrame:
    return pd.read_excel(file_or_path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_axis([c.lower().strip() for c in df.columns], axis=1, copy=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Debés incluir: {REQUIRED_COLUMNS} y opcional {OPTIONAL_COLUMNS}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    df["concepto"] = df["concepto"].apply(_normalize_text)
    df["descripcion"] = df["descripcion"].apply(_normalize_text) if "descripcion" in df.columns else ""

    df["monto"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0.0).abs()
    df = df[df["monto"] != 0]

    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month

    return df.reset_index(drop=True)

def cluster_concepts(df: pd.DataFrame, n_clusters: int = 8):
    corpus = (df["concepto"].fillna("") + " " + df["descripcion"].fillna("")).values

    stopwords = [
        "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
        "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o",
        "este","sí","porque","esta","entre","cuando","muy","sin","sobre","también",
        "me","hasta","hay","donde","quien","desde","todo","nos","durante","todos",
        "uno","les","ni","contra","otros","ese","eso","ante","ellos","e","esto",
        "mí","antes","qué","unos","yo","otro","otras","otra","él","ella","ellos",
        "ellas","usted","ustedes","mi","tu","te","ti","su"
    ]

    vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=2)
    X = vectorizer.fit_transform(corpus)

    n_clusters = max(2, min(n_clusters, max(2, X.shape[0] // 20)))
    # IA: KMeans usado para clustering de conceptos similares
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["categoria_auto"] = labels

    inv_vocab = {i: t for t, i in vectorizer.vocabulary_.items()}

    try:
        centers = kmeans.cluster_centers_
        names = []
        for i in range(centers.shape[0]):
            idx = centers[i].argsort()[-5:][::-1]
            terms = [inv_vocab.get(j, "") for j in idx]
            names.append(", ".join([t for t in terms if t][:2]) or f"cat_{i}")
        df["categoria_nombre"] = df["categoria_auto"].map({i: names[i] for i in range(len(names))})
    except Exception:
        df["categoria_nombre"] = "categoria"

    return df, vectorizer, kmeans

def monthly_pivot(df: pd.DataFrame, use_names: bool = True) -> pd.DataFrame:
    col = "categoria_nombre" if use_names and "categoria_nombre" in df.columns else "categoria_auto"
    pv = df.pivot_table(index="mes", columns=col, values="monto", aggfunc="sum", fill_value=0.0).sort_index()
    pv.index = pd.PeriodIndex(pv.index, freq="M").astype(str)
    pv["total"] = pv.sum(axis=1)
    return pv

def build_supervised_dataset(pv: pd.DataFrame):
    X = pv.drop(columns=["total"], errors="ignore")
    y = pv["total"].shift(-1)
    return X.iloc[:-1].copy(), y.iloc[:-1].copy()

def rolling_stats(df: pd.DataFrame):
    by_day = df.groupby("fecha")["monto"].sum().rename("gasto_diario")
    by_cat = df.groupby("categoria_nombre")["monto"].sum().sort_values(ascending=False)
    return by_day.to_frame(), by_cat.to_frame(name="gasto_por_categoria")
