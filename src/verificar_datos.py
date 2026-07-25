from pathlib import Path
import polars as pl

RAIZ = Path(__file__).resolve().parent.parent
RUTA_EVENTOS = RAIZ / "data" / "processed" / "eventos_limpios.parquet"

eventos = pl.scan_parquet(str(RUTA_EVENTOS))

print("=== Schema ===")
print(eventos.collect_schema())

print("\n=== Conteo total y por tipo de evento ===")
print(eventos.select(pl.len()).collect().item(), "filas totales")
print(
    eventos.group_by("event")
    .agg(pl.len().alias("cantidad"))
    .sort("cantidad", descending=True)
    .collect()
)

print("\n=== Usuarios y productos únicos ===")
resumen = eventos.select([
    pl.col("visitorid").n_unique().alias("usuarios_unicos"),
    pl.col("itemid").n_unique().alias("productos_unicos"),
]).collect()
print(resumen)

print("\n=== Distribución de peso_implicito ===")
print(
    eventos.group_by("peso_implicito")
    .agg(pl.len().alias("cantidad"))
    .sort("peso_implicito")
    .collect()
)

print("\n=== Muestra de filas ===")
print(eventos.head(5).collect())