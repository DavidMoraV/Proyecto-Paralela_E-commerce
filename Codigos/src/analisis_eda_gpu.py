"""
analisis_eda_gpu.py
Módulo 2 (M2) — Clustering GPU (Entrega 3)
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

Extiende el análisis exploratorio de M2 (Entrega 2, CPU/Scikit-Learn) con
una versión acelerada por GPU usando RAPIDS cuML, y compara ambas para
identificar a partir de qué volumen de datos la GPU compensa su overhead
de transferencia frente a CPU.

Archivo separado de `analisis_eda.py` a propósito: así el módulo original
(ya usado por M3, M4 y el dashboard) no se modifica ni se arriesga a
romperse por una dependencia (cuML) que solo existe en el entorno de Colab.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M2_clustering_gpu")

COLUMNAS_FEATURES = [
    "total_interacciones", "productos_distintos", "peso_total",
    "views", "carritos", "compras",
]


# ---------------------------------------------------------------------------
# 1. Clustering en GPU (cuML) -- misma interfaz que segmentar_usuarios (CPU)
# ---------------------------------------------------------------------------
def segmentar_usuarios_gpu(
    usuarios_features: pl.DataFrame,
    n_clusters: int = 6,
    columnas: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pl.DataFrame, float]:
    """Equivalente en GPU de `segmentar_usuarios` (analisis_eda.py, CPU).
    Aplica log1p + estandarización + K-Means, pero con cuML sobre GPU.
    Devuelve el DataFrame con la columna 'cluster' añadida, y el tiempo de
    entrenamiento en segundos.
    """
    import cudf
    from cuml.cluster import KMeans as KMeansGPU
    from cuml.preprocessing import StandardScaler as StandardScalerGPU

    columnas = columnas or COLUMNAS_FEATURES
    logger.info("Segmentando %s usuarios en %d clusters (GPU)", f"{usuarios_features.height:,}", n_clusters)

    features_log = usuarios_features.select(columnas).with_columns(
        [pl.col(c).log1p().alias(c) for c in columnas]
    )
    df_gpu = cudf.from_pandas(features_log.to_pandas())
    X_scaled = StandardScalerGPU().fit_transform(df_gpu)

    t0 = time.perf_counter()
    modelo = KMeansGPU(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = modelo.fit_predict(X_scaled)
    tiempo = time.perf_counter() - t0

    clusters_np = clusters.to_numpy() if hasattr(clusters, "to_numpy") else np.asarray(clusters)
    resultado = usuarios_features.with_columns(pl.Series("cluster", clusters_np))
    logger.info("Clustering GPU completado en %.3f s", tiempo)
    return resultado, tiempo


def segmentar_usuarios_cpu_referencia(
    usuarios_features: pl.DataFrame,
    n_clusters: int = 6,
    columnas: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pl.DataFrame, float]:
    """Misma lógica que `segmentar_usuarios` de analisis_eda.py, pero
    devolviendo también el tiempo medido, para comparar directamente contra
    la versión GPU con el mismo formato de retorno.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    columnas = columnas or COLUMNAS_FEATURES
    logger.info("Segmentando %s usuarios en %d clusters (CPU)", f"{usuarios_features.height:,}", n_clusters)

    features_log = usuarios_features.select(columnas).with_columns(
        [pl.col(c).log1p().alias(c) for c in columnas]
    )
    X_scaled = StandardScaler().fit_transform(features_log.to_pandas())

    t0 = time.perf_counter()
    modelo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = modelo.fit_predict(X_scaled)
    tiempo = time.perf_counter() - t0

    resultado = usuarios_features.with_columns(pl.Series("cluster", clusters))
    logger.info("Clustering CPU completado en %.3f s", tiempo)
    return resultado, tiempo


# ---------------------------------------------------------------------------
# 2. Benchmark GPU vs CPU a distintos tamaños de dataset
# ---------------------------------------------------------------------------
def _generar_features_sinteticas(n_filas: int, n_columnas: int = 6, semilla: int = 42) -> pd.DataFrame:
    """Genera una tabla de features sintética con la misma forma que las
    features reales de usuario (positivas, distribución long-tail similar a
    conteos de interacciones), para poder medir el comportamiento de GPU vs
    CPU a volúmenes (100K/1M/10M) que superan la cantidad real de usuarios
    del dataset (1.4M).
    """
    rng = np.random.default_rng(semilla)
    datos = rng.lognormal(mean=1.5, sigma=1.2, size=(n_filas, n_columnas))
    return pd.DataFrame(datos, columns=[f"f{i}" for i in range(n_columnas)])


def benchmark_kmeans_gpu_vs_cpu(
    tamanos: list[int] | None = None,
    n_clusters: int = 6,
    n_repeticiones: int = 3,
) -> pd.DataFrame:
    """Mide el tiempo de K-Means (CPU/Scikit-Learn vs GPU/cuML) sobre
    datasets sintéticos de tamaño creciente, para identificar a partir de
    qué volumen la GPU compensa su overhead de transferencia de datos.

    Se usan datos sintéticos (no el dataset real de usuarios) porque el
    objetivo es explorar el UMBRAL de tamaño, y el dataset real (1.4M
    usuarios) no alcanza los 10M de filas que pide explorar la rúbrica.
    """
    import cudf
    from cuml.cluster import KMeans as KMeansGPU
    from sklearn.cluster import KMeans as KMeansCPU

    tamanos = tamanos or [100_000, 1_000_000, 10_000_000]
    filas = []

    for n in tamanos:
        logger.info("Generando dataset sintetico de %s filas", f"{n:,}")
        df_pd = _generar_features_sinteticas(n)

        # --- CPU ---
        tiempos_cpu = []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            KMeansCPU(n_clusters=n_clusters, n_init=3, random_state=42).fit_predict(df_pd)
            tiempos_cpu.append(time.perf_counter() - t0)

        # --- GPU ---
        df_gpu = cudf.from_pandas(df_pd)
        tiempos_gpu = []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            KMeansGPU(n_clusters=n_clusters, n_init=3, random_state=42).fit_predict(df_gpu)
            tiempos_gpu.append(time.perf_counter() - t0)

        t_cpu, t_gpu = float(np.mean(tiempos_cpu)), float(np.mean(tiempos_gpu))
        filas.append({
            "n_filas": n,
            "tiempo_cpu_s": t_cpu,
            "tiempo_gpu_s": t_gpu,
            "speedup_gpu": round(t_cpu / t_gpu, 2) if t_gpu > 0 else float("inf"),
            "gpu_mas_rapida": t_gpu < t_cpu,
        })
        logger.info(
            "n=%s -> CPU %.3fs, GPU %.3fs, speedup=%.2fx",
            f"{n:,}", t_cpu, t_gpu, t_cpu / t_gpu if t_gpu > 0 else float("inf"),
        )

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3. Reducción dimensional (UMAP, GPU) para visualización 2D
# ---------------------------------------------------------------------------
def reducir_umap_gpu(usuarios_features: pl.DataFrame, columnas: list[str] | None = None,
                      n_muestra: int | None = 50_000, semilla: int = 42) -> pd.DataFrame:
    """Reduce las features de usuario a 2 dimensiones con UMAP (GPU, cuML),
    para visualizar los clusters en un plano. Se limita a una muestra
    (`n_muestra`) porque UMAP es costoso incluso en GPU sobre 1.4M+ puntos,
    y el objetivo es visualización, no un resultado exhaustivo.
    """
    import cudf
    from cuml.manifold import UMAP

    columnas = columnas or COLUMNAS_FEATURES
    df = usuarios_features
    if n_muestra and df.height > n_muestra:
        df = df.sample(n=n_muestra, seed=semilla)

    features_log = df.select(columnas).with_columns([pl.col(c).log1p().alias(c) for c in columnas])
    df_gpu = cudf.from_pandas(features_log.to_pandas())

    logger.info("Corriendo UMAP (GPU) sobre %s puntos", f"{df.height:,}")
    embedding = UMAP(n_components=2, random_state=semilla).fit_transform(df_gpu)
    embedding_pd = embedding.to_pandas() if hasattr(embedding, "to_pandas") else pd.DataFrame(embedding)
    embedding_pd.columns = ["umap_x", "umap_y"]

    resultado = df.select(["visitorid", "cluster"]).to_pandas().reset_index(drop=True)
    resultado = pd.concat([resultado, embedding_pd.reset_index(drop=True)], axis=1)
    return resultado
