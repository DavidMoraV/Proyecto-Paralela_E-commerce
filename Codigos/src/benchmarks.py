"""
benchmarks.py
Módulo 4 (M4) — Computación Paralela y Análisis de Rendimiento
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

Responsable: HPC Engineer (M4, Entrega 2)

Scripts de benchmarking reutilizables para medir speedup y eficiencia de las
etapas del pipeline (ETL de M1, clustering de M2, entrenamiento de M3), y
para perfilar el entrenamiento del modelo e identificar cuellos de botella.

Protocolo de medición (ver "Gestión de problemas" en el informe):
- Cada configuración se corre con 1 repetición de calentamiento (warm-up),
  descartada, seguida de N repeticiones que sí se promedian.
- Se reporta media, desviación estándar, mínimo y máximo — no solo el
  promedio — para que la variabilidad entre corridas quede documentada
  en vez de oculta.
"""
from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M4_benchmarks")


# ---------------------------------------------------------------------------
# 1. Protocolo de medición: warm-up + repeticiones
# ---------------------------------------------------------------------------
@dataclass
class ResultadoMedicion:
    """Resultado de medir una función repetidamente, con warm-up descartado."""
    etiqueta: str
    tiempos: list[float] = field(default_factory=list)

    @property
    def media(self) -> float:
        return sum(self.tiempos) / len(self.tiempos) if self.tiempos else float("nan")

    @property
    def desviacion(self) -> float:
        if len(self.tiempos) < 2:
            return 0.0
        m = self.media
        return (sum((t - m) ** 2 for t in self.tiempos) / (len(self.tiempos) - 1)) ** 0.5

    @property
    def minimo(self) -> float:
        return min(self.tiempos) if self.tiempos else float("nan")

    @property
    def maximo(self) -> float:
        return max(self.tiempos) if self.tiempos else float("nan")

    def to_dict(self) -> dict:
        return {
            "etiqueta": self.etiqueta,
            "media_s": round(self.media, 4),
            "std_s": round(self.desviacion, 4),
            "min_s": round(self.minimo, 4),
            "max_s": round(self.maximo, 4),
            "n_repeticiones": len(self.tiempos),
        }


def medir_tiempo_repetido(
    func,
    *args,
    etiqueta: str = "medicion",
    n_repeticiones: int = 3,
    warm_up: int = 1,
    **kwargs,
) -> ResultadoMedicion:
    """Mide el tiempo de `func(*args, **kwargs)` con calentamiento (warm-up)
    descartado y N repeticiones que sí se promedian.

    El warm-up es necesario porque la primera ejecución de cualquier función
    en Python suele incluir costos que no se repiten después (import
    perezoso, compilación JIT de librerías como Polars/NumPy, cachés de
    sistema operativo fríos), lo que infla artificialmente la primera
    medición si no se descarta.
    """
    logger.info("Midiendo '%s': %d warm-up + %d repeticiones", etiqueta, warm_up, n_repeticiones)

    for _ in range(warm_up):
        func(*args, **kwargs)

    resultado = ResultadoMedicion(etiqueta=etiqueta)
    for i in range(n_repeticiones):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        resultado.tiempos.append(t1 - t0)
        logger.info("  Repetición %d/%d: %.4f s", i + 1, n_repeticiones, t1 - t0)

    return resultado


import dask
import dask.dataframe as dd

from pipeline_datos import detectar_separador, resolver_ruta


def calcular_speedup_eficiencia(df: pd.DataFrame, columna_workers: str, columna_tiempo: str) -> pd.DataFrame:
    """Agrega columnas de speedup y eficiencia a una tabla de resultados,
    usando la fila de 1 worker como referencia secuencial.
    """
    df = df.copy()
    baseline = df.loc[df[columna_workers] == 1, columna_tiempo].iloc[0]
    df["speedup"] = (baseline / df[columna_tiempo]).round(3)
    df["eficiencia"] = (df["speedup"] / df[columna_workers]).round(3)
    return df


def _cargar_y_limpiar_dask(ruta: Path, num_workers: int, blocksize: str = "16MB") -> int:
    """Carga y limpia events.csv con Dask, forzando el scheduler a usar
    exactamente `num_workers` hilos. Devuelve el número de filas resultantes
    (fuerza el cómputo completo, no solo construir el grafo de tareas).

    `blocksize` controla el tamaño de cada partición al leer el CSV, y por
    lo tanto cuántas particiones reales se generan — un archivo de 94MB con
    blocksize="16MB" genera solo ~6 particiones, lo que limita el paralelismo
    disponible aunque se pidan más workers (ver 'Gestión de problemas').
    """
    ruta = resolver_ruta(ruta)
    sep = detectar_separador(ruta)

    df = dd.read_csv(ruta, sep=sep, dtype={"itemid": "float64"}, blocksize=blocksize)
    if sep == ";":
        df["timestamp"] = df["timestamp"].astype(str).str.replace(",", ".").astype(float).astype("int64")
    df = df.drop_duplicates()
    df = df.dropna(subset=["itemid", "visitorid"])
    df = df[df["event"].isin(["view", "addtocart", "transaction"])]

    with dask.config.set(scheduler="threads", num_workers=num_workers):
        resultado = df.compute()
    return len(resultado)


def contar_particiones(ruta: Path, blocksize: str = "16MB") -> int:
    """Cuenta cuántas particiones reales genera Dask para un archivo dado,
    con un blocksize determinado. Útil para diagnosticar si el número de
    workers pedido excede las particiones disponibles.
    """
    ruta = resolver_ruta(ruta)
    sep = detectar_separador(ruta)
    df = dd.read_csv(ruta, sep=sep, blocksize=blocksize)
    return df.npartitions


def benchmark_etl_workers(
    ruta_eventos: Path,
    lista_workers: list[int] | None = None,
    n_repeticiones: int = 3,
    warm_up: int = 1,
    blocksize: str = "16MB",
) -> pd.DataFrame:
    """Mide el tiempo de carga+limpieza de events.csv con Dask, variando el
    número de workers (1, 2, 4, 8 por defecto), con warm-up y repeticiones.
    Devuelve una tabla con speedup y eficiencia respecto a 1 worker (secuencial).
    """
    lista_workers = lista_workers or [1, 2, 4, 8]
    n_particiones = contar_particiones(ruta_eventos, blocksize)
    logger.info("blocksize=%s -> %d particiones reales generadas", blocksize, n_particiones)
    filas = []

    for w in lista_workers:
        if w > n_particiones:
            logger.warning(
                "Se pidieron %d workers pero el archivo solo tiene %d particiones "
                "(blocksize=%s) -- el paralelismo real está limitado a %d.",
                w, n_particiones, blocksize, n_particiones,
            )
        resultado = medir_tiempo_repetido(
            _cargar_y_limpiar_dask, ruta_eventos, w, blocksize,
            etiqueta=f"ETL Dask ({w} workers, {n_particiones} particiones)",
            n_repeticiones=n_repeticiones, warm_up=warm_up,
        )
        fila = resultado.to_dict()
        fila["workers"] = w
        fila["particiones"] = n_particiones
        filas.append(fila)

    df = pd.DataFrame(filas)
    return calcular_speedup_eficiencia(df, columna_workers="workers", columna_tiempo="media_s")


# ---------------------------------------------------------------------------
# 2. Profiling del entrenamiento (identificación de cuellos de botella)
# ---------------------------------------------------------------------------
def perfilar_funcion(func, *args, top_n: int = 15, **kwargs) -> tuple[object, pd.DataFrame]:
    """Corre `func(*args, **kwargs)` bajo cProfile y devuelve (resultado, tabla
    de las top_n funciones por tiempo acumulado), para identificar cuellos de
    botella sin adivinar.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    resultado = func(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)

    filas = []
    for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
        archivo, linea, nombre = func_key
        filas.append({
            "funcion": nombre,
            "archivo": Path(archivo).name,
            "linea": linea,
            "llamadas": nc,
            "tiempo_propio_s": round(tt, 4),
            "tiempo_acumulado_s": round(ct, 4),
        })

    tabla = pd.DataFrame(filas).sort_values("tiempo_acumulado_s", ascending=False).head(top_n)
    return resultado, tabla.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Guardado de resultados
# ---------------------------------------------------------------------------
def benchmark_clustering_threads(
    X_scaled,
    lista_hilos: list[int] | None = None,
    n_clusters: int = 6,
    n_repeticiones: int = 3,
    warm_up: int = 1,
) -> pd.DataFrame:
    """Mide el tiempo de entrenamiento de K-Means (etapa de clustering de M2)
    variando el número de hilos permitidos para las librerías numéricas
    subyacentes (OpenMP/BLAS), usando threadpoolctl — mismo mecanismo que ya
    se usó para detectar el problema de oversubscription en M3.
    """
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    lista_hilos = lista_hilos or [1, 2, 4]
    filas = []

    for n_hilos in lista_hilos:
        def _entrenar_kmeans():
            with threadpool_limits(limits=n_hilos):
                KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)

        resultado = medir_tiempo_repetido(
            _entrenar_kmeans,
            etiqueta=f"K-Means ({n_hilos} hilos)",
            n_repeticiones=n_repeticiones, warm_up=warm_up,
        )
        fila = resultado.to_dict()
        fila["workers"] = n_hilos
        filas.append(fila)

    df = pd.DataFrame(filas)
    return calcular_speedup_eficiencia(df, columna_workers="workers", columna_tiempo="media_s")


def benchmark_entrenamiento_als_threads(
    matriz_train,
    lista_hilos: list[int] | None = None,
    factors: int = 32,
    iterations: int = 10,
    n_repeticiones: int = 2,
    warm_up: int = 1,
) -> pd.DataFrame:
    """Mide el tiempo de entrenamiento ALS (etapa de M3) variando el número
    de hilos permitidos para BLAS, para cuantificar el efecto de
    oversubscription detectado durante el desarrollo de M3.
    """
    from implicit.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits

    lista_hilos = lista_hilos or [1, 2, 4, 8]
    filas = []

    for n_hilos in lista_hilos:
        def _entrenar():
            modelo = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=42)
            with threadpool_limits(limits=n_hilos, user_api="blas"):
                modelo.fit(matriz_train)

        resultado = medir_tiempo_repetido(
            _entrenar,
            etiqueta=f"ALS ({n_hilos} hilos BLAS)",
            n_repeticiones=n_repeticiones, warm_up=warm_up,
        )
        fila = resultado.to_dict()
        fila["workers"] = n_hilos
        filas.append(fila)

    df = pd.DataFrame(filas)
    return calcular_speedup_eficiencia(df, columna_workers="workers", columna_tiempo="media_s")


def guardar_resultados(df: pd.DataFrame, nombre: str, carpeta_resultados: Path) -> Path:
    carpeta_resultados.mkdir(exist_ok=True, parents=True)
    destino = carpeta_resultados / nombre
    df.to_csv(destino, index=False)
    logger.info("Resultados guardados en %s", destino)
    return destino
