"""
analisis_eda.py
Módulo 2 (M2) — Análisis Exploratorio y Estadístico
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

Responsable: Ana María Ramírez (M2, Entrega 2)

Refactorización a funciones reutilizables del notebook original de EDA y
segmentación de usuarios, para que M3 pueda importar directamente la
construcción de features y la carga de segmentos sin repetir código.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M2_analisis_eda")

COLUMNAS_FEATURES = [
    "total_interacciones", "productos_distintos", "peso_total",
    "views", "carritos", "compras",
]


def construir_features_usuario(eventos: pl.LazyFrame) -> pl.DataFrame:
    """Construye la matriz de features por usuario a partir de eventos limpios.

    Cada fila representa un usuario; las columnas resumen su actividad
    (interacciones totales, productos distintos, peso implícito acumulado,
    y conteo por tipo de evento).
    """
    logger.info("Construyendo features de usuario")
    return (
        eventos
        .group_by("visitorid")
        .agg([
            pl.len().alias("total_interacciones"),
            pl.col("itemid").n_unique().alias("productos_distintos"),
            pl.col("peso_implicito").sum().alias("peso_total"),
            (pl.col("event") == "view").sum().alias("views"),
            (pl.col("event") == "addtocart").sum().alias("carritos"),
            (pl.col("event") == "transaction").sum().alias("compras"),
        ])
        .collect()
    )


def segmentar_usuarios(
    usuarios_features: pl.DataFrame,
    n_clusters: int = 6,
    columnas: list[str] | None = None,
    random_state: int = 42,
) -> pl.DataFrame:
    """Aplica log1p + StandardScaler + K-Means sobre las features de usuario.

    Devuelve el DataFrame original con una columna 'cluster' añadida.
    """
    columnas = columnas or COLUMNAS_FEATURES
    logger.info("Segmentando %s usuarios en %d clusters", f"{usuarios_features.height:,}", n_clusters)

    features_log = usuarios_features.select(columnas).with_columns(
        [pl.col(c).log1p().alias(c) for c in columnas]
    )
    X_scaled = StandardScaler().fit_transform(features_log.to_pandas())

    modelo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = modelo.fit_predict(X_scaled)

    return usuarios_features.with_columns(pl.Series("cluster", clusters))


def exportar_segmentos(usuarios_features: pl.DataFrame, carpeta_processed: Path) -> Path:
    """Exporta el resultado de la segmentación a user_segments.parquet."""
    destino = carpeta_processed / "user_segments.parquet"
    usuarios_features.write_parquet(destino)
    logger.info("Segmentos exportados a %s", destino)
    return destino


def cargar_segmentos(carpeta_processed: Path) -> pl.DataFrame:
    """Punto de entrada para M3: carga directamente user_segments.parquet ya
    generado por M2, sin repetir el clustering."""
    return pl.read_parquet(carpeta_processed / "user_segments.parquet")


def perfil_clusters(usuarios_features: pl.DataFrame) -> pl.DataFrame:
    """Promedios por cluster, útil para interpretar cada segmento."""
    return (
        usuarios_features
        .group_by("cluster")
        .agg([pl.mean(c).alias(c) for c in COLUMNAS_FEATURES])
        .sort("cluster")
    )
