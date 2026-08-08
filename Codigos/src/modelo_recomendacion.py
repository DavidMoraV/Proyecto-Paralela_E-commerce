
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M3_modelo_recomendacion")



# 1. Construcción de la matriz de interacciones

@dataclass
class MatrizInteracciones:
    """Contenedor con la matriz dispersa usuario-producto y los mapeos de ID
    originales (visitorid/itemid) a índices contiguos (0..n-1), necesarios
    para indexar la matriz dispersa.
    """
    matriz: csr_matrix
    visitorid_a_idx: dict[int, int]
    idx_a_visitorid: dict[int, int]
    itemid_a_idx: dict[int, int]
    idx_a_itemid: dict[int, int]


def construir_matriz_interacciones(eventos: pl.DataFrame) -> MatrizInteracciones:
    """Construye la matriz usuario-producto (dispersa) usando peso_implicito
    como valor de confianza, sumado si un usuario interactuó varias veces con
    el mismo producto.
    """
    logger.info("Construyendo matriz de interacciones usuario-producto")

    agregado = (
        eventos
        .group_by(["visitorid", "itemid"])
        .agg(pl.col("peso_implicito").sum().alias("peso"))
    )

    visitores = agregado["visitorid"].unique().sort().to_list()
    items = agregado["itemid"].unique().sort().to_list()

    visitorid_a_idx = {v: i for i, v in enumerate(visitores)}
    itemid_a_idx = {it: i for i, it in enumerate(items)}

    filas = agregado["visitorid"].replace_strict(visitorid_a_idx, return_dtype=pl.Int64).to_numpy()
    columnas = agregado["itemid"].replace_strict(itemid_a_idx, return_dtype=pl.Int64).to_numpy()
    valores = agregado["peso"].to_numpy().astype(np.float32)

    matriz = coo_matrix(
        (valores, (filas, columnas)),
        shape=(len(visitores), len(items)),
    ).tocsr()

    logger.info(
        "Matriz construida: %s usuarios x %s productos (%s interacciones no nulas)",
        f"{matriz.shape[0]:,}", f"{matriz.shape[1]:,}", f"{matriz.nnz:,}",
    )

    return MatrizInteracciones(
        matriz=matriz,
        visitorid_a_idx=visitorid_a_idx,
        idx_a_visitorid={i: v for v, i in visitorid_a_idx.items()},
        itemid_a_idx=itemid_a_idx,
        idx_a_itemid={i: it for it, i in itemid_a_idx.items()},
    )


# 2. División train/test (leave-one-out por usuario, basada en tiempo)

def dividir_train_test(eventos: pl.DataFrame, min_interacciones: int = 2) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separa, para cada usuario con al menos `min_interacciones` eventos, su
    interacción MÁS RECIENTE como test y el resto como train. Los usuarios con
    una sola interacción quedan solo en train (no se pueden evaluar).

    Esta estrategia (leave-one-out temporal) es estándar para evaluar
    sistemas de recomendación con feedback implícito.
    """
    logger.info("Dividiendo train/test (leave-one-out temporal)")

    eventos_con_rank = eventos.with_columns(
        pl.col("timestamp").rank(method="ordinal", descending=True).over("visitorid").alias("rank_reciente")
    )

    conteo = eventos.group_by("visitorid").agg(pl.len().alias("n"))
    usuarios_evaluables = set(conteo.filter(pl.col("n") >= min_interacciones)["visitorid"].to_list())

    test = eventos_con_rank.filter(
        (pl.col("rank_reciente") == 1) & (pl.col("visitorid").is_in(usuarios_evaluables))
    ).drop("rank_reciente")

    train = eventos_con_rank.filter(
        (pl.col("rank_reciente") != 1) | (~pl.col("visitorid").is_in(usuarios_evaluables))
    ).drop("rank_reciente")

    logger.info(
        "Train: %s filas | Test: %s filas (%s usuarios evaluables)",
        f"{train.height:,}", f"{test.height:,}", f"{len(usuarios_evaluables):,}",
    )
    return train, test


# 3. Modelo baseline: ALS

class ModeloALS:

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        self.modelo = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        self._entrenado = False

    def entrenar(self, matriz_train: csr_matrix) -> float:
        """Entrena el modelo. Devuelve el tiempo de entrenamiento en segundos.

        Se limita OpenBLAS a 1 hilo durante el entrenamiento (vía
        threadpoolctl) para evitar oversubscription: sin esto, OpenBLAS
        lanza su propio pool de hilos que compite por los mismos núcleos
        que ya usa el paralelismo interno de Implicit, degradando el
        rendimiento en vez de mejorarlo (ver 'Gestión de problemas').
        """
        logger.info(
            "Entrenando ALS (factors=%d, iterations=%d) sobre %s x %s",
            self.modelo.factors, self.modelo.iterations, *matriz_train.shape,
        )
        t0 = time.perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            self.modelo.fit(matriz_train)
        t1 = time.perf_counter()
        self._entrenado = True
        logger.info("Entrenamiento completado en %.2f segundos", t1 - t0)
        return t1 - t0

    def recomendar(self, usuario_idx: int, matriz_train: csr_matrix, n: int = 10) -> list[int]:
        """Top-N recomendaciones (índices de producto) para un usuario, excluyendo
        productos con los que ya interactuó en train."""
        if not self._entrenado:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a .entrenar() primero.")
        ids, _ = self.modelo.recommend(
            usuario_idx, matriz_train[usuario_idx], N=n, filter_already_liked_items=True
        )
        return list(ids)


# 4. Métricas de evaluación (ranking)

def exportar_recomendaciones_muestra(
    modelo: ModeloALS,
    matriz_train: csr_matrix,
    mi: MatrizInteracciones,
    n_usuarios: int = 100,
    n_recomendaciones: int = 10,
    semilla: int = 42,
) -> pl.DataFrame:
    """Genera recomendaciones para una muestra aleatoria de usuarios y las
    devuelve en formato largo (una fila por usuario-recomendación), lista
    para exportar a CSV y ser consumida por el dashboard de M5 sin que este
    tenga que cargar ni reentrenar el modelo completo.
    """
    rng = np.random.default_rng(semilla)
    usuarios_disponibles = list(mi.visitorid_a_idx.keys())
    muestra_visitorid = rng.choice(usuarios_disponibles, size=min(n_usuarios, len(usuarios_disponibles)), replace=False)
    muestra_idx = np.array([mi.visitorid_a_idx[v] for v in muestra_visitorid])

    ids_recomendados, scores = modelo.modelo.recommend(
        muestra_idx, matriz_train[muestra_idx], N=n_recomendaciones, filter_already_liked_items=True
    )

    filas = []
    for visitorid, ids_fila, scores_fila in zip(muestra_visitorid, ids_recomendados, scores):
        for rank, (item_idx, score) in enumerate(zip(ids_fila, scores_fila), start=1):
            filas.append({
                "visitorid": int(visitorid),
                "rank": rank,
                "itemid": mi.idx_a_itemid[int(item_idx)],
                "score": float(score),
            })

    return pl.DataFrame(filas)


def guardar_recomendaciones_muestra(df: pl.DataFrame, carpeta_resultados) -> None:
    destino = carpeta_resultados / "recomendaciones_muestra.csv"
    df.write_csv(destino)
    logger.info("Recomendaciones de muestra guardadas en %s", destino)


def evaluar_modelo(
    modelo: ModeloALS,
    matriz_train: csr_matrix,
    test: pl.DataFrame,
    mi: MatrizInteracciones,
    k: int = 10,
    max_usuarios: int | None = None,
) -> dict[str, float]:
    logger.info("Evaluando modelo con K=%d", k)

    filas_test = test.select(["visitorid", "itemid"]).to_numpy()
    if max_usuarios:
        filas_test = filas_test[:max_usuarios]

    hits, ap_scores, ndcg_scores = [], [], []

    for visitorid, itemid in filas_test:
        u_idx = mi.visitorid_a_idx.get(int(visitorid))
        i_idx = mi.itemid_a_idx.get(int(itemid))
        if u_idx is None or i_idx is None:
            continue  # producto o usuario no visto en train

        recomendados = modelo.recomendar(u_idx, matriz_train, n=k)

        if i_idx in recomendados:
            rank = recomendados.index(i_idx) + 1  # 1-indexado
            hits.append(1)
            ap_scores.append(1.0 / rank)
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0)
            ap_scores.append(0.0)
            ndcg_scores.append(0.0)

    n_evaluados = len(hits)
    return {
        "usuarios_evaluados": n_evaluados,
        f"hit_rate@{k}": float(np.mean(hits)) if n_evaluados else 0.0,
        f"map@{k}": float(np.mean(ap_scores)) if n_evaluados else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if n_evaluados else 0.0,
    }


def evaluar_modelo_batch(
    modelo: ModeloALS,
    matriz_train: csr_matrix,
    test: pl.DataFrame,
    mi: MatrizInteracciones,
    k: int = 10,
    max_usuarios: int | None = None,
    tam_lote: int = 5000,
) -> dict[str, float]:
    """Igual que `evaluar_modelo`, pero usa el modo batch nativo de Implicit
    (recomienda para miles de usuarios a la vez, vectorizado) en vez de un
    loop de Python usuario por usuario. Permite evaluar el conjunto de test
    completo en segundos en vez de minutos/horas.
    """
    logger.info("Evaluando modelo (batch) con K=%d", k)

    filas_test = test.select(["visitorid", "itemid"]).to_numpy()
    if max_usuarios:
        filas_test = filas_test[:max_usuarios]

    # Vectorizar la traducción visitorid/itemid -> índices, descartando
    # usuarios/productos no vistos en train.
    u_idx_arr = np.array([mi.visitorid_a_idx.get(int(v)) for v, _ in filas_test])
    i_idx_arr = np.array([mi.itemid_a_idx.get(int(it)) for _, it in filas_test])
    validos = (u_idx_arr != None) & (i_idx_arr != None)  # noqa: E711
    u_idx_arr = u_idx_arr[validos].astype(int)
    i_idx_arr = i_idx_arr[validos].astype(int)

    hits, ap_scores, ndcg_scores = [], [], []

    for inicio in range(0, len(u_idx_arr), tam_lote):
        lote_usuarios = u_idx_arr[inicio: inicio + tam_lote]
        lote_items_reales = i_idx_arr[inicio: inicio + tam_lote]

        ids_recomendados, _ = modelo.modelo.recommend(
            lote_usuarios,
            matriz_train[lote_usuarios],
            N=k,
            filter_already_liked_items=True,
        )

        for recomendados, item_real in zip(ids_recomendados, lote_items_reales):
            recomendados = list(recomendados)
            if item_real in recomendados:
                rank = recomendados.index(item_real) + 1
                hits.append(1)
                ap_scores.append(1.0 / rank)
                ndcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                hits.append(0)
                ap_scores.append(0.0)
                ndcg_scores.append(0.0)

        logger.info("Evaluados %s / %s usuarios", f"{min(inicio + tam_lote, len(u_idx_arr)):,}", f"{len(u_idx_arr):,}")

    n_evaluados = len(hits)
    return {
        "usuarios_evaluados": n_evaluados,
        f"hit_rate@{k}": float(np.mean(hits)) if n_evaluados else 0.0,
        f"map@{k}": float(np.mean(ap_scores)) if n_evaluados else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if n_evaluados else 0.0,
    }
