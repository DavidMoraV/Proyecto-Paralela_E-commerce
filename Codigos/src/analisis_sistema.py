"""
analisis_sistema.py
Módulo 4 (M4) — Análisis de Rendimiento del Sistema Integrado (Entrega 3)
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

A diferencia de la Entrega 2 (donde cada etapa se benchmarkeó por separado),
este módulo mide el sistema COMPLETO de punta a punta: ETL -> clustering ->
entrenamiento -> inferencia, además de balance de carga y costos de
comunicación entre workers de Dask (métricas de sistemas distribuidos que
no se habían cuantificado hasta ahora).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M4_analisis_sistema")


# ---------------------------------------------------------------------------
# 1. Pipeline completo, de punta a punta, con tiempo por etapa
# ---------------------------------------------------------------------------
def ejecutar_pipeline_completo(
    ruta_eventos: Path,
    ruta_category_tree: Path,
    ruta_item_props1: Path,
    ruta_item_props2: Path,
    n_clusters: int = 6,
    factors: int = 64,
    n_usuarios_inferencia: int = 1000,
) -> dict:
    """Corre el sistema completo (M1 -> M2 -> M3 entrenamiento -> M3
    inferencia) y devuelve el tiempo de cada etapa, tal como se
    experimentaría en un despliegue real de principio a fin.
    """
    from pipeline_datos import (
        ingestar_eventos, limpiar_eventos, transformar_eventos,
        ingestar_category_tree, limpiar_category_tree,
    )
    from analisis_eda import construir_features_usuario, segmentar_usuarios
    from modelo_recomendacion import (
        construir_matriz_interacciones, dividir_train_test, ModeloALS,
    )

    tiempos = {}

    # --- M1: Ingesta + limpieza ---
    t0 = time.perf_counter()
    eventos = limpiar_eventos(ingestar_eventos(ruta_eventos))
    eventos = transformar_eventos(eventos).collect()
    tiempos["M1_ingesta_limpieza"] = time.perf_counter() - t0
    logger.info("M1 (ingesta+limpieza): %.2f s -- %s filas", tiempos["M1_ingesta_limpieza"], f"{eventos.height:,}")

    # --- M2: Clustering ---
    t0 = time.perf_counter()
    features = construir_features_usuario(eventos.lazy())
    segmentos = segmentar_usuarios(features, n_clusters=n_clusters)
    tiempos["M2_clustering"] = time.perf_counter() - t0
    logger.info("M2 (clustering): %.2f s -- %s usuarios", tiempos["M2_clustering"], f"{segmentos.height:,}")

    # --- M3: Entrenamiento ---
    t0 = time.perf_counter()
    train, test = dividir_train_test(eventos)
    mi = construir_matriz_interacciones(train)
    modelo = ModeloALS(factors=factors, iterations=15)
    modelo.entrenar(mi.matriz)
    tiempos["M3_entrenamiento"] = time.perf_counter() - t0
    logger.info("M3 (entrenamiento): %.2f s", tiempos["M3_entrenamiento"])

    # --- M3: Inferencia (batch de recomendaciones, escenario de producción) ---
    t0 = time.perf_counter()
    usuarios_disponibles = list(mi.visitorid_a_idx.values())
    muestra = np.array(usuarios_disponibles[:min(n_usuarios_inferencia, len(usuarios_disponibles))])
    modelo.modelo.recommend(muestra, mi.matriz[muestra], N=10, filter_already_liked_items=True)
    tiempos["M3_inferencia_batch"] = time.perf_counter() - t0
    logger.info("M3 (inferencia, %d usuarios): %.4f s", len(muestra), tiempos["M3_inferencia_batch"])

    tiempos["TOTAL"] = sum(v for k, v in tiempos.items() if k != "TOTAL")
    return tiempos


# ---------------------------------------------------------------------------
# 2. Latencia de inferencia según tamaño de lote (para "prueba de carga")
# ---------------------------------------------------------------------------
def benchmark_latencia_inferencia(
    modelo, matriz_train, mi, tamanos_lote: list[int] | None = None, n_repeticiones: int = 5,
) -> pd.DataFrame:
    """Mide la latencia de generar recomendaciones para distintos tamaños de
    lote (1 usuario = latencia individual, lotes grandes = throughput de
    producción). Relevante para el escenario de picos de tráfico.
    """
    tamanos_lote = tamanos_lote or [1, 10, 100, 1000]
    usuarios_disponibles = np.array(list(mi.visitorid_a_idx.values()))
    filas = []

    for n in tamanos_lote:
        n_real = min(n, len(usuarios_disponibles))
        muestra = usuarios_disponibles[:n_real]

        tiempos = []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            modelo.modelo.recommend(muestra, matriz_train[muestra], N=10, filter_already_liked_items=True)
            tiempos.append(time.perf_counter() - t0)

        tiempos = np.array(tiempos)
        filas.append({
            "tamano_lote": n_real,
            "latencia_media_s": tiempos.mean(),
            "latencia_media_ms": tiempos.mean() * 1000,
            "latencia_por_usuario_ms": (tiempos.mean() / n_real) * 1000,
            "throughput_usuarios_por_s": n_real / tiempos.mean(),
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3. Balance de carga y costos de comunicación (Dask distributed)
# ---------------------------------------------------------------------------
def medir_balance_y_comunicacion(ruta_eventos: Path, n_workers: int = 4) -> dict:
    """Mide, sobre un clúster local de Dask distributed, cómo se reparte el
    trabajo entre workers (balance de carga) y cuánto tiempo se gasta
    transfiriendo datos entre ellos (costo de comunicación) al correr la
    limpieza de eventos -- la etapa con más probabilidad de requerir
    shuffle (deduplicación) entre particiones.
    """
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster, get_task_stream
    from pipeline_datos import detectar_separador, resolver_ruta

    ruta_eventos = resolver_ruta(ruta_eventos)
    sep = detectar_separador(ruta_eventos)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, dashboard_address=None)
    client = Client(cluster)
    try:
        with get_task_stream(client=client) as ts:
            df = dd.read_csv(ruta_eventos, sep=sep, dtype={"itemid": "float64"}, blocksize="16MB")
            df = df.drop_duplicates()  # etapa con shuffle real entre particiones
            df = df.dropna(subset=["itemid", "visitorid"])
            df.compute()

        registros = ts.data  # lista de dicts con 'worker', 'startstops', etc.

        tiempo_ocupado_por_worker: dict[str, float] = {}
        tiempo_transferencia_total = 0.0

        for r in registros:
            worker = r.get("worker", "desconocido")
            for evento in r.get("startstops", []):
                duracion = evento["stop"] - evento["start"]
                if evento["action"] == "compute":
                    tiempo_ocupado_por_worker[worker] = tiempo_ocupado_por_worker.get(worker, 0.0) + duracion
                elif evento["action"] == "transfer":
                    tiempo_transferencia_total += duracion

        valores = list(tiempo_ocupado_por_worker.values())
        balance = {
            "tiempo_ocupado_por_worker": tiempo_ocupado_por_worker,
            "media_s": float(np.mean(valores)) if valores else 0.0,
            "std_s": float(np.std(valores)) if valores else 0.0,
            "coef_variacion": float(np.std(valores) / np.mean(valores)) if valores and np.mean(valores) > 0 else 0.0,
            "tiempo_transferencia_total_s": tiempo_transferencia_total,
            "n_workers": n_workers,
            "n_tareas_registradas": len(registros),
        }
        logger.info(
            "Balance de carga: coef. de variación=%.3f (0=perfecto) | Comunicación: %.3fs transferidos entre workers",
            balance["coef_variacion"], balance["tiempo_transferencia_total_s"],
        )
        return balance
    finally:
        client.close()
        cluster.close()
