"""
Evaluación de modelos de recomendación: NDCG@10 y MAP@10.
Usa el esquema leave-one-out: cada usuario en test tiene un único
ítem retenido (su interacción más reciente).
"""
from pathlib import Path
import time

import numpy as np
import polars as pl
import scipy.sparse as sp


def evaluar_modelo(modelo, matriz_train: sp.csr_matrix, test: pl.DataFrame,
                    k: int = 10, batch_size: int = 2000) -> dict:
    """Calcula NDCG@k y MAP@k sobre el conjunto de test (leave-one-out).

    Para cada usuario, se generan las top-k recomendaciones (excluyendo
    ítems ya vistos en train) y se revisa si el ítem retenido aparece,
    y en qué posición.
    """
    usuarios_test = test["usuario_idx"].to_numpy()
    items_verdaderos = test["item_idx"].to_numpy()

    ndcg_scores = []
    ap_scores = []
    aciertos = 0  # hit rate: cuántas veces el ítem verdadero aparece en el top-k

    t0 = time.perf_counter()
    n_usuarios = len(usuarios_test)

    for inicio in range(0, n_usuarios, batch_size):
        fin = min(inicio + batch_size, n_usuarios)
        lote_usuarios = usuarios_test[inicio:fin]
        lote_items_verdaderos = items_verdaderos[inicio:fin]

        ids_recomendados, _ = modelo.modelo.recommend(
            lote_usuarios,
            matriz_train[lote_usuarios],
            N=k,
            filter_already_liked_items=True,
        )

        for recomendados, item_real in zip(ids_recomendados, lote_items_verdaderos):
            posiciones = np.where(recomendados == item_real)[0]
            if len(posiciones) > 0:
                rank = posiciones[0] + 1  # posición 1-indexada
                ndcg_scores.append(1.0 / np.log2(rank + 1))
                ap_scores.append(1.0 / rank)
                aciertos += 1
            else:
                ndcg_scores.append(0.0)
                ap_scores.append(0.0)

        if (inicio // batch_size) % 20 == 0:
            print(f"  Procesados {fin:,}/{n_usuarios:,} usuarios...")

    tiempo_total = time.perf_counter() - t0

    return {
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "map_at_10": float(np.mean(ap_scores)),
        "hit_rate_at_10": aciertos / n_usuarios,
        "n_usuarios_evaluados": n_usuarios,
        "tiempo_evaluacion_seg": tiempo_total,
    }

def evaluar_por_segmento(modelo, matriz_train: sp.csr_matrix, test: pl.DataFrame,
                          train: pl.DataFrame, umbral_interacciones: int = 5,
                          k: int = 10) -> dict:
    """Evalúa por separado usuarios 'fríos' (< umbral interacciones en train)
    vs usuarios con historial suficiente (>= umbral)."""
    interacciones_train = (
        train.group_by("usuario_idx").agg(pl.len().alias("n_interacciones"))
    )
    usuarios_calidos = set(
        interacciones_train.filter(pl.col("n_interacciones") >= umbral_interacciones)
        ["usuario_idx"].to_list()
    )

    test_calido = test.filter(pl.col("usuario_idx").is_in(usuarios_calidos))
    test_frio = test.filter(~pl.col("usuario_idx").is_in(usuarios_calidos))

    print(f"\nUsuarios con >= {umbral_interacciones} interacciones en train: "
          f"{test_calido.height:,} en test")
    print(f"Usuarios con < {umbral_interacciones} interacciones en train: "
          f"{test_frio.height:,} en test")

    resultados = {}
    if test_calido.height > 0:
        resultados["segmento_calido"] = evaluar_modelo(modelo, matriz_train, test_calido, k=k)
    if test_frio.height > 0:
        resultados["segmento_frio"] = evaluar_modelo(modelo, matriz_train, test_frio, k=k)
    return resultados
def evaluar_popularidad(train: pl.DataFrame, test: pl.DataFrame, k: int = 10) -> dict:
    """Baseline trivial: recomienda siempre los k ítems más frecuentes en train,
    sin personalización. Sirve como piso de comparación para ALS."""
    top_items = (
        train.group_by("item_idx")
        .agg(pl.len().alias("cantidad"))
        .sort("cantidad", descending=True)
        .head(k)["item_idx"]
        .to_numpy()
    )

    items_verdaderos = test["item_idx"].to_numpy()
    ndcg_scores = []
    ap_scores = []
    aciertos = 0

    for item_real in items_verdaderos:
        posiciones = np.where(top_items == item_real)[0]
        if len(posiciones) > 0:
            rank = posiciones[0] + 1
            ndcg_scores.append(1.0 / np.log2(rank + 1))
            ap_scores.append(1.0 / rank)
            aciertos += 1
        else:
            ndcg_scores.append(0.0)
            ap_scores.append(0.0)

    return {
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "map_at_10": float(np.mean(ap_scores)),
        "hit_rate_at_10": aciertos / len(items_verdaderos),
        "n_usuarios_evaluados": len(items_verdaderos),
    }

if __name__ == "__main__":
    from modelo_recomendacion import BaselineModel

    RAIZ = Path(__file__).resolve().parent.parent
    RUTA_MODELO_DATOS = RAIZ / "data" / "processed" / "modelo"
    RUTA_MODELOS_GUARDADOS = RAIZ / "resultados" / "modelos"

    print("Cargando matriz de train y datos de test...")
    matriz_train = sp.load_npz(RUTA_MODELO_DATOS / "matriz_train.npz")
    test = pl.read_parquet(RUTA_MODELO_DATOS / "test.parquet")

    print("Cargando modelo baseline entrenado...")
    baseline = BaselineModel()
    baseline.modelo = baseline.modelo.load(str(RUTA_MODELOS_GUARDADOS / "baseline_als.npz"))

    print(f"\nEvaluando sobre {test.height:,} usuarios de test...")
    metricas = evaluar_modelo(baseline, matriz_train, test, k=10)

    print("\n=== Resultados Baseline (ALS) ===")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor}")

    import json
    RUTA_RESULTADOS = RAIZ / "resultados"
    with open(RUTA_RESULTADOS / "metricas_baseline.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"\nGuardado en {RUTA_RESULTADOS / 'metricas_baseline.json'}")

    print("\nEvaluando por segmento (cold-start vs. usuarios con historial)...")
    train = pl.read_parquet(RUTA_MODELO_DATOS / "train.parquet")
    metricas_segmento = evaluar_por_segmento(baseline, matriz_train, test, train, umbral_interacciones=5)

    print("\n=== Resultados por segmento ===")
    for segmento, m in metricas_segmento.items():
        print(f"\n{segmento}:")
        for nombre, valor in m.items():
            print(f"  {nombre}: {valor}")

    with open(RUTA_RESULTADOS / "metricas_baseline_por_segmento.json", "w") as f:
        json.dump(metricas_segmento, f, indent=2)

    print("\nEvaluando baseline de popularidad (sin personalización)...")
    metricas_popularidad = evaluar_popularidad(train, test, k=10)
    print("\n=== Resultados Popularidad (baseline trivial) ===")
    for nombre, valor in metricas_popularidad.items():
        print(f"{nombre}: {valor}")

    with open(RUTA_RESULTADOS / "metricas_popularidad.json", "w") as f:
        json.dump(metricas_popularidad, f, indent=2)