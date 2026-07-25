from pathlib import Path
import polars as pl

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO_DATOS = RAIZ / "data" / "processed" / "modelo"

train = pl.read_parquet(RUTA_MODELO_DATOS / "train.parquet")

interacciones_por_usuario = (
    train.group_by("usuario_idx")
    .agg(pl.len().alias("n_interacciones"))
)

print("=== Distribución de interacciones por usuario en TRAIN ===")
print(interacciones_por_usuario["n_interacciones"].describe())

print("\n=== Percentiles clave ===")
for p in [50, 75, 90, 95, 99]:
    valor = interacciones_por_usuario["n_interacciones"].quantile(p / 100)
    print(f"Percentil {p}: {valor}")

usuarios_1_interaccion = interacciones_por_usuario.filter(
    pl.col("n_interacciones") == 1
).height
total_usuarios = interacciones_por_usuario.height
print(f"\nUsuarios con exactamente 1 interacción en train: {usuarios_1_interaccion:,} "
      f"({100*usuarios_1_interaccion/total_usuarios:.1f}%)")