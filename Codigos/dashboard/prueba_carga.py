
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M5_prueba_carga")

ENDPOINT = "/_dash-update-component"



# Construcción de payloads (mismo formato que usa el cliente JS de Dash)

def payload_cambiar_tab(valor_tab: str) -> dict:
    return {
        "output": "contenido-tab.children",
        "outputs": {"id": "contenido-tab", "property": "children"},
        "inputs": [{"id": "tabs", "property": "value", "value": valor_tab}],
        "changedPropIds": ["tabs.value"],
    }


def payload_seleccionar_usuario(visitorid: int) -> dict:
    return {
        "output": "tabla-recomendaciones.children",
        "outputs": {"id": "tabla-recomendaciones", "property": "children"},
        "inputs": [{"id": "selector-usuario", "property": "value", "value": visitorid}],
        "changedPropIds": ["selector-usuario.value"],
    }



# Ejecución de una sola solicitud (una "visita" de un usuario)

def _una_solicitud(url_base: str, payload: dict, timeout: float = 30.0) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url_base}{ENDPOINT}", json=payload, timeout=timeout)
        latencia = time.perf_counter() - t0
        return {"latencia_s": latencia, "status": resp.status_code, "error": resp.status_code != 200}
    except requests.RequestException as e:
        latencia = time.perf_counter() - t0
        return {"latencia_s": latencia, "status": None, "error": True, "excepcion": str(e)}



# Prueba de carga: N usuarios concurrentes, cada uno con una solicitud "real"
# (usuario aleatorio distinto, simulando que no se puede cachear la respuesta)

def prueba_carga_recomendaciones(
    url_base: str,
    usuarios_disponibles: list[int],
    n_concurrentes: int,
    n_solicitudes_totales: int | None = None,
) -> dict:
    """Simula `n_concurrentes` usuarios pidiendo recomendaciones al mismo
    tiempo (cada uno con un visitorid distinto, elegido al azar). Devuelve
    latencia (media, p50, p95, p99, max) y tasa de error.
    """
    n_solicitudes_totales = n_solicitudes_totales or n_concurrentes
    payloads = [
        payload_seleccionar_usuario(random.choice(usuarios_disponibles))
        for _ in range(n_solicitudes_totales)
    ]

    t0 = time.perf_counter()
    resultados = []
    with ThreadPoolExecutor(max_workers=n_concurrentes) as executor:
        futuros = [executor.submit(_una_solicitud, url_base, p) for p in payloads]
        for f in as_completed(futuros):
            resultados.append(f.result())
    tiempo_total = time.perf_counter() - t0

    latencias = np.array([r["latencia_s"] for r in resultados])
    n_errores = sum(1 for r in resultados if r["error"])

    return {
        "n_concurrentes": n_concurrentes,
        "n_solicitudes": n_solicitudes_totales,
        "tiempo_total_s": tiempo_total,
        "throughput_req_por_s": n_solicitudes_totales / tiempo_total,
        "latencia_media_ms": float(latencias.mean() * 1000),
        "latencia_p50_ms": float(np.percentile(latencias, 50) * 1000),
        "latencia_p95_ms": float(np.percentile(latencias, 95) * 1000),
        "latencia_p99_ms": float(np.percentile(latencias, 99) * 1000),
        "latencia_max_ms": float(latencias.max() * 1000),
        "n_errores": n_errores,
        "tasa_error": n_errores / n_solicitudes_totales,
    }


def simular_pico_trafico(
    url_base: str,
    usuarios_disponibles: list[int],
    niveles_concurrencia: list[int] | None = None,
) -> pd.DataFrame:
    """Corre la prueba de carga en niveles crecientes de concurrencia,
    simulando la rampa de un pico de tráfico (ej. apertura de Black Friday).
    """
    niveles_concurrencia = niveles_concurrencia or [1, 10, 50, 100, 200]
    filas = []
    for n in niveles_concurrencia:
        logger.info("Simulando %d usuarios concurrentes...", n)
        resultado = prueba_carga_recomendaciones(url_base, usuarios_disponibles, n_concurrentes=n)
        logger.info(
            "  %d concurrentes -> latencia media %.1fms, p95 %.1fms, throughput %.1f req/s, errores %.1f%%",
            n, resultado["latencia_media_ms"], resultado["latencia_p95_ms"],
            resultado["throughput_req_por_s"], resultado["tasa_error"] * 100,
        )
        filas.append(resultado)
    return pd.DataFrame(filas)
