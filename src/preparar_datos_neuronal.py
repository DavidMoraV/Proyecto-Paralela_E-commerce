"""
Construye pares (usuario, item, etiqueta) con muestreo negativo,
a partir del train ya generado en el Paso 5, para entrenar el modelo NCF.
"""
from pathlib import Path

import numpy as np
import polars as pl

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO_DATOS = RAIZ / "data" / "processed" / "modelo"

N_NEGATIVOS_POR_POSITIVO = 4  # estándar en el paper original de NCF


def generar_pares_entrenamiento(train: pl.DataFrame, n_items: int,
                                 n_negativos: int = N_NEGATIVOS_POR_POSITIVO,
                                 semilla: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)

    positivos_usuario = train["usuario_idx"].to_numpy()
    positivos_item = train["item_idx"].to_numpy()
    n_positivos = len(positivos_usuario)

    # Set de pares vistos por usuario, para no generar un "negativo" que en
    # realidad sí fue visto (evita falsos negativos)
    vistos_por_usuario: dict[int, set] = {}
    for u, i in zip(positivos_usuario, positivos_item):
        vistos_por_usuario.setdefault(u, set()).add(i)

    usuarios_final = np.empty(n_positivos * (1 + n_negativos), dtype=np.int64)
    items_final = np.empty(n_positivos * (1 + n_negativos), dtype=np.int64)
    etiquetas_final = np.empty(n_positivos * (1 + n_negativos), dtype=np.float32)

    idx = 0
    for u, i in zip(positivos_usuario, positivos_item):
        usuarios_final[idx] = u
        items_final[idx] = i
        etiquetas_final[idx] = 1.0
        idx += 1

        vistos = vistos_por_usuario[u]
        negativos_generados = 0
        intentos = 0
        while negativos_generados < n_negativos and intentos < n_negativos * 20:
            candidato = rng.integers(0, n_items)
            intentos += 1
            if candidato not in vistos:
                usuarios_final[idx] = u
                items_final[idx] = candidato
                etiquetas_final[idx] = 0.0
                idx += 1
                negativos_generados += 1

    return usuarios_final[:idx], items_final[:idx], etiquetas_final[:idx]


if __name__ == "__main__":
    import time

    print("Cargando train...")
    train = pl.read_parquet(RUTA_MODELO_DATOS / "train.parquet")
    n_items = train["item_idx"].max() + 1

    print(f"Generando pares con {N_NEGATIVOS_POR_POSITIVO} negativos por positivo...")
    t0 = time.perf_counter()
    usuarios, items, etiquetas = generar_pares_entrenamiento(train, n_items)
    print(f"Listo en {time.perf_counter() - t0:.1f}s")

    print(f"Total de pares: {len(usuarios):,} "
          f"({(etiquetas == 1).sum():,} positivos, {(etiquetas == 0).sum():,} negativos)")

    np.savez(
        RUTA_MODELO_DATOS / "pares_entrenamiento_neural.npz",
        usuarios=usuarios, items=items, etiquetas=etiquetas,
    )
    print("Guardado en", RUTA_MODELO_DATOS / "pares_entrenamiento_neural.npz")