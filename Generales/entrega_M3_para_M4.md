

### 1. Resumen

Modelo baseline de recomendación (ALS / filtrado colaborativo con feedback implícito), entrenado y evaluado sobre el dataset completo de RetailRocket:

| Dato | Valor |
|---|---|
| Eventos de entrada (desde M1) | 2,755,641 |
| Usuarios (desde M2) | 1,407,580 |
| División train/test | Leave-one-out temporal: 2,349,652 train / 405,989 test |
| Matriz usuario-producto | 1,407,580 × 227,081 (1,930,255 interacciones no nulas, densidad 0.0006%) |
| Usuarios evaluados | 397,571 (97.9% del test evaluable) |
| Hit Rate@10 | 0.0266 |
| MAP@10 | 0.0124 |
| NDCG@10 | 0.0157 |

### 2. Tiempos por etapa (ya medidos  punto de partida para M4)

| Etapa | Tiempo medido | Notas para profiling |
|---|---|---|
| Carga de datos (M1 + M2, Parquet) | ~2 s | Rápido; no es un cuello de botella relevante |
| División train/test (`dividir_train_test`) | ~1.6 s | Usa `.rank().over("visitorid")` en Polars sobre 2.75M filas |
| Construcción de matriz dispersa (`construir_matriz_interacciones`) | ~1.5 s | `group_by` + conversión a `coo_matrix`/`csr_matrix` |
| **Entrenamiento ALS (15 iteraciones, factors=64)** | **102.9 s** | **La etapa más costosa con diferencia — candidata principal para profiling detallado** |
| Evaluación batch (397,571 usuarios, lotes de 5,000) | ~280 s (≈4.7 min) | Segunda etapa más costosa; paralelizable por lotes |

**Recomendación para M4:** concentrar el profiling (`cProfile`, `line_profiler`) en las etapas de entrenamiento y evaluación — juntas representan más del 95% del tiempo total de ejecución del pipeline de M3.

### 3. Problema de rendimiento ya identificado (relevante directamente para M4)

Durante el entrenamiento aparece esta advertencia real, no simulada:

```
OpenBLAS is configured to use 12 threads. It is highly recommended to disable
its internal threadpool... Having OpenBLAS use a threadpool can lead to
severe performance issues here.
```

Esto es un caso real de **oversubscription**: OpenBLAS lanza su propio pool de 12 hilos que compite por los mismos núcleos que ya usa el paralelismo interno de Implicit. Ya se mitigó parcialmente envolviendo el entrenamiento con `threadpool_limits` (`src/modelo_recomendacion.py`), pero la advertencia persiste porque se dispara al *construir* el modelo, no al entrenarlo.

**Esto es, literalmente, un experimento de speedup/eficiencia ya armado para M4:** comparar el tiempo de entrenamiento con `OPENBLAS_NUM_THREADS=1` vs sin restricción (12 hilos) es una medición directa de los efectos de oversubscription — cuadra exactamente con el objetivo del módulo ("identificar cuellos de botella y aplicar optimizaciones").

Sugerencia de experimento concreto para M4:

```python
import os, time
for n_threads in [1, 2, 4, 8, 12]:
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    # reiniciar kernel entre corridas (la variable debe fijarse antes de importar numpy)
    # medir tiempo de modelo.entrenar(matriz_train) para cada configuración
```

### 4. experimentos de speedup/escalabilidad sugeridos para M4

- **Factors del modelo ALS** (32 / 64 / 128 / 256): medir tiempo de entrenamiento vs calidad (NDCG@10) — relación costo/beneficio de aumentar la dimensionalidad de los embeddings.
- **Tamaño de lote en la evaluación batch** (`tam_lote`: 1,000 / 5,000 / 20,000): medir tiempo total de evaluación por configuración — trade-off entre paralelismo y overhead de coordinación, mismo patrón que ya se observó con Dask en M1 (a mayor lote, menos overhead relativo, hasta cierto punto).
- **Iteraciones de ALS** (5 / 15 / 30): tiempo vs convergencia de las métricas de calidad — para saber si 15 iteraciones (el valor actual) es un punto razonable o si hay margen de ahorro.

### 5. herramientas

| Archivo | Contenido | Ubicación |
|---|---|---|
| `src/modelo_recomendacion.py` | Módulo de M3, incluye `ModeloALS.entrenar()` que ya devuelve el tiempo de entrenamiento en segundos, listo para instrumentar | `Codigos/` |
| `resultados/metricas_modelo_baseline.csv` | Hit Rate@10, MAP@10, NDCG@10 sobre 397,571 usuarios | `Codigos/resultados/` |
| `resultados/hit_rate_por_cluster.csv` | Rendimiento del modelo desglosado por los 6 clusters de M2 | `Codigos/resultados/` |
| `processed/eventos_limpios.parquet`, `processed/user_segments.parquet` | Insumos de M1/M2 que M4 puede reutilizar directamente sin reprocesar nada | `Codigos/processed/` |

### 6. Nota de integración

`ModeloALS.entrenar()` ya está diseñado para ser fácilmente instrumentable: recibe la matriz de entrenamiento y devuelve el tiempo transcurrido, así que M4 puede importar directamente `from modelo_recomendacion import ModeloALS, construir_matriz_interacciones` y variar hiperparámetros (factors, iterations) o la configuración de hilos sin tocar el código de M3, solo llamando la función con distintos parámetros en un loop de benchmarking.
