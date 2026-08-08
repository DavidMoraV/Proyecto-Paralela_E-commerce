
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M1_pipeline")


def detectar_separador(ruta: Path) -> str:
    """Detecta si el CSV usa ',' o ';' como separador."""
    with open(ruta, encoding="utf-8") as f:
        primera_linea = f.readline()
    return ";" if primera_linea.count(";") > primera_linea.count(",") else ","


def resolver_ruta(ruta_csv: Path) -> Path:
    """Si el .csv no existe pero sí existe el .xlsx equivalente, usa ese."""
    if ruta_csv.exists():
        return ruta_csv
    alterna = ruta_csv.with_suffix(".xlsx")
    if alterna.exists():
        logger.info("No se encontró %s, usando %s en su lugar", ruta_csv.name, alterna.name)
        return alterna
    raise FileNotFoundError(f"No se encontró ni {ruta_csv.name} ni {alterna.name}")


def ingestar_eventos(ruta: Path) -> pl.LazyFrame:
    """Lee events.csv, normalizando separador y timestamp si vino de Excel."""
    ruta = resolver_ruta(ruta)
    logger.info("Ingestando eventos desde %s", ruta)
    separador = detectar_separador(ruta)
    if separador == ";":
        logger.warning(
            "Separador ';' detectado en %s: archivo reformateado por Excel. "
            "El timestamp puede tener precisión reducida.",
            ruta.name,
        )

    crudo = pl.scan_csv(
        ruta,
        separator=separador,
        schema_overrides={
            "visitorid": pl.Int64,
            "event": pl.Utf8,
            "itemid": pl.Float64,
            "transactionid": pl.Float64,
        },
    )

    if separador == ";":
        crudo = crudo.with_columns(
            pl.col("timestamp").str.replace(",", ".").cast(pl.Float64).cast(pl.Int64)
        )
    else:
        crudo = crudo.with_columns(pl.col("timestamp").cast(pl.Int64))

    return crudo


def ingestar_category_tree(ruta: Path) -> pl.LazyFrame:
    """Lee category_tree.csv (árbol de categorías hijo-padre)."""
    ruta = resolver_ruta(ruta)
    separador = detectar_separador(ruta)
    return pl.scan_csv(ruta, separator=separador)


def leer_item_properties_individual(ruta: Path) -> pl.LazyFrame:
    """Lee un archivo item_properties_part*, aceptando .csv o .xlsx."""
    ruta = resolver_ruta(ruta)
    if ruta.suffix.lower() == ".xlsx":
        logger.info("Leyendo %s como Excel (motor nativo Polars)", ruta.name)
        crudo = pl.read_excel(ruta, schema_overrides={"itemid": pl.Int64}).lazy()
        return crudo.with_columns(pl.col("timestamp").cast(pl.Int64))

    separador = detectar_separador(ruta)
    crudo = pl.scan_csv(
        ruta,
        separator=separador,
        schema_overrides={"itemid": pl.Int64, "property": pl.Utf8, "value": pl.Utf8},
    )
    if separador == ";":
        crudo = crudo.with_columns(
            pl.col("timestamp").str.replace(",", ".").cast(pl.Float64).cast(pl.Int64)
        )
    else:
        crudo = crudo.with_columns(pl.col("timestamp").cast(pl.Int64))
    return crudo


def ingestar_item_properties(ruta1: Path, ruta2: Path) -> pl.LazyFrame:
    """Lee y concatena las dos partes de item_properties."""
    lf1 = leer_item_properties_individual(ruta1)
    lf2 = leer_item_properties_individual(ruta2)
    return pl.concat([lf1, lf2], how="vertical")


def limpiar_eventos(eventos: pl.LazyFrame) -> pl.LazyFrame:
    """Limpieza de events.csv: duplicados, nulos, valores de 'event' válidos."""
    filas_antes = eventos.select(pl.len()).collect().item()
    n_ts_unicos = eventos.select(pl.col("timestamp").n_unique()).collect().item()
    perdida_precision = filas_antes > 0 and n_ts_unicos / filas_antes < 0.01

    limpio = (
        eventos
        .unique()
        .filter(pl.col("itemid").is_not_null() & pl.col("visitorid").is_not_null())
        .filter(pl.col("event").is_in(["view", "addtocart", "transaction"]))
        .with_columns(pl.col("itemid").cast(pl.Int64))
    )

    filas_despues = limpio.select(pl.len()).collect().item()
    eliminadas = filas_antes - filas_despues
    logger.info(
        "Limpieza de eventos: %s -> %s filas (%s eliminadas, %.2f%%)",
        f"{filas_antes:,}", f"{filas_despues:,}", f"{eliminadas:,}",
        100 * eliminadas / filas_antes if filas_antes else 0,
    )

    if perdida_precision and filas_antes and eliminadas / filas_antes > 0.02:
        logger.warning(
            "ALERTA DE CALIDAD: %s filas (%.1f%%) eliminadas como 'duplicados', pero el "
            "timestamp solo tiene %s valores únicos en %s filas.",
            f"{eliminadas:,}", 100 * eliminadas / filas_antes, f"{n_ts_unicos:,}", f"{filas_antes:,}",
        )
    return limpio


def limpiar_category_tree(categorias: pl.LazyFrame) -> pl.LazyFrame:
    return categorias.unique(subset=["categoryid"])


def transformar_eventos(eventos: pl.LazyFrame) -> pl.LazyFrame:
    """peso_implicito (view=1, addtocart=3, transaction=5), fecha, hora_del_dia, dia_semana."""
    pesos = {"view": 1, "addtocart": 3, "transaction": 5}
    return eventos.with_columns([
        pl.from_epoch("timestamp", time_unit="ms").alias("fecha"),
        pl.col("event").replace_strict(pesos, default=1, return_dtype=pl.Int32).alias("peso_implicito"),
    ]).with_columns([
        pl.col("fecha").dt.hour().alias("hora_del_dia"),
        pl.col("fecha").dt.weekday().alias("dia_semana"),
    ])


def cargar_eventos_procesados(carpeta_processed: Path) -> pl.LazyFrame:
    """Punto de entrada para M2/M3: carga directamente el Parquet ya limpio de M1,
    sin repetir ingesta/limpieza."""
    return pl.scan_parquet(str(carpeta_processed / "eventos_limpios.parquet"))
