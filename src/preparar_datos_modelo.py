from pathlib import Path
import polars as pl
import scipy.sparse as sp
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RUTA_EVENTOS = RAIZ / "data" / "processed" / "eventos_limpios.parquet"
RUTA_SALIDA = RAIZ / "data" / "processed" / "modelo"
RUTA_SALIDA.mkdir(exist_ok=True, parents=True)


def cargar_eventos() -> pl.DataFrame:
    """Carga los eventos limpios, ya con peso_implicito calculado por M1."""
    return pl.read_parquet(str(RUTA_EVENTOS)).select(
        ["visitorid", "itemid", "timestamp", "peso_implicito"]
    )


def construir_indices(eventos: pl.DataFrame) -> tuple[pl.DataFrame, dict, dict]:
    """Mapea visitorid/itemid (IDs originales, dispersos) a índices densos 0..N,
    que es lo que espera una matriz dispersa de scipy."""
    usuarios_unicos = eventos["visitorid"].unique().sort()
    items_unicos = eventos["itemid"].unique().sort()

    mapa_usuario = {uid: idx for idx, uid in enumerate(usuarios_unicos.to_list())}
    mapa_item = {iid: idx for idx, iid in enumerate(items_unicos.to_list())}

    eventos = eventos.with_columns([
        pl.col("visitorid").replace_strict(mapa_usuario).alias("usuario_idx"),
        pl.col("itemid").replace_strict(mapa_item).alias("item_idx"),
    ])
    return eventos, mapa_usuario, mapa_item


def split_train_test(eventos: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Leave-one-out temporal: la interacción MÁS RECIENTE de cada usuario va a test,
    el resto a train. Usuarios con una sola interacción total van ENTERAMENTE a train
    (no se puede evaluar "predecir la siguiente" sin historial previo)."""
    eventos = eventos.with_columns(
        pl.col("timestamp")
        .count()
        .over("usuario_idx")
        .alias("n_interacciones")
    )
    eventos = eventos.with_columns(
        pl.col("timestamp")
        .rank(method="ordinal", descending=True)
        .over("usuario_idx")
        .alias("orden_reciente")
    )

    test = eventos.filter(
        (pl.col("orden_reciente") == 1) & (pl.col("n_interacciones") > 1)
    )
    train = eventos.filter(
        (pl.col("orden_reciente") != 1) | (pl.col("n_interacciones") == 1)
    )

    return train, test


def construir_matriz_dispersa(datos: pl.DataFrame, n_usuarios: int, n_items: int) -> sp.csr_matrix:
    """Matriz usuario-item dispersa, usando peso_implicito como valor de confianza."""
    filas = datos["usuario_idx"].to_numpy()
    columnas = datos["item_idx"].to_numpy()
    valores = datos["peso_implicito"].to_numpy().astype(np.float32)
    return sp.csr_matrix((valores, (filas, columnas)), shape=(n_usuarios, n_items))


if __name__ == "__main__":
    print("Cargando eventos...")
    eventos = cargar_eventos()

    print("Construyendo índices densos...")
    eventos, mapa_usuario, mapa_item = construir_indices(eventos)
    n_usuarios = len(mapa_usuario)
    n_items = len(mapa_item)
    print(f"Usuarios: {n_usuarios:,} | Items: {n_items:,}")

    print("Separando train/test (leave-one-out temporal)...")
    train, test = split_train_test(eventos)
    print(f"Train: {train.height:,} filas | Test: {test.height:,} filas")

    print("Construyendo matrices dispersas...")
    matriz_train = construir_matriz_dispersa(train, n_usuarios, n_items)
    matriz_test = construir_matriz_dispersa(test, n_usuarios, n_items)

    print(f"Matriz train: {matriz_train.shape}, {matriz_train.nnz:,} valores no-cero")
    print(f"Matriz test:  {matriz_test.shape}, {matriz_test.nnz:,} valores no-cero")

    print("Guardando matrices y mapeos...")
    sp.save_npz(RUTA_SALIDA / "matriz_train.npz", matriz_train)
    sp.save_npz(RUTA_SALIDA / "matriz_test.npz", matriz_test)
    train.write_parquet(RUTA_SALIDA / "train.parquet")
    test.write_parquet(RUTA_SALIDA / "test.parquet")

    import json
    with open(RUTA_SALIDA / "mapa_usuario.json", "w") as f:
        json.dump({str(k): v for k, v in mapa_usuario.items()}, f)
    with open(RUTA_SALIDA / "mapa_item.json", "w") as f:
        json.dump({str(k): v for k, v in mapa_item.items()}, f)

    print("Listo. Archivos en:", RUTA_SALIDA)