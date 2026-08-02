## M2 — Clustering GPU (cuML) vs CPU (Scikit-Learn): Resultados Finales (Entrega 3)

### Funcionalidad

Se implementó clustering K-Means ($k=6$) sobre las features de usuario (interacciones totales, productos distintos, peso implícito, conteo por tipo de evento) en dos variantes: CPU (Scikit-Learn, ya usada en la Entrega 2) y GPU (RAPIDS cuML), con la misma metodología de preprocesamiento (`log1p` + estandarización) en ambas. Se corrió sobre el dataset real completo (1,407,580 usuarios) y sobre datasets sintéticos de tamaño creciente (100K, 1M, 10M filas) para explorar el comportamiento de escalabilidad.

### Eficiencia y escalabilidad: benchmark aislado de K-Means

Se midió el tiempo de entrenamiento de K-Means, sin preprocesamiento adicional, sobre datasets sintéticos de tamaño creciente.

| Filas | CPU | GPU | Speedup GPU |
|---|---|---|---|
| 100,000 | 0.52 s | 0.32 s | 1.63x |
| 1,000,000 | 3.68 s | 2.59 s | 1.42x |
| 10,000,000 | 39.49 s | 25.82 s | 1.53x |

A diferencia de la comparativa de M1 (Pandas/Polars/Dask/cuDF, Entrega 3), donde la ventaja de GPU aparecía solo a partir de cierto volumen de datos, **en K-Means la GPU resulta más rápida ya desde el tamaño más pequeño evaluado** (100K filas), con un speedup moderado y estable en todo el rango (1.4x–1.6x). Esto es consistente con la naturaleza del algoritmo: K-Means es iterativo y con cómputo matricial repetido en cada iteración, por lo que amortiza el overhead de transferencia CPU→GPU más rápido que una operación de una sola pasada como la lectura de un CSV.

### Hallazgo principal: el pipeline completo en producción invierte el resultado

Se comparó también el pipeline real de segmentación (preprocesamiento + clustering) sobre los 1,407,580 usuarios reales del dataset.

| Configuración | Tiempo |
|---|---|
| CPU (Scikit-Learn, pipeline completo) | 13.64 s |
| GPU (cuML, pipeline completo) | 15.90 s |
| **Speedup GPU** | **0.86x (GPU más lenta)** |

Este resultado contradice, en apariencia, el benchmark aislado de la sección anterior (donde GPU ganaba con un volumen de datos comparable, ~1M filas). La explicación no está en el tamaño de los datos, sino en **qué se está midiendo**:

1. **El benchmark aislado mide solo K-Means; el pipeline real mide preprocesamiento + K-Means.** La función usada en producción (`segmentar_usuarios_gpu`) aplica `log1p`, convierte los datos a `cudf.DataFrame`, y corre `StandardScaler` de cuML antes de entrenar el modelo — cada uno de estos pasos adicionales implica su propia transferencia de datos y lanzamiento de kernels en GPU, overhead que el benchmark aislado (que entrena K-Means directamente sobre datos ya en GPU, sin preprocesamiento) no paga.
2. **Diferencia en `n_init`.** El benchmark aislado usa `n_init=3`; el pipeline real usa `n_init=10` (para mantener consistencia metodológica con la Entrega 2), lo que multiplica el número de entrenamientos completos ejecutados.
3. **Composición particular del dataset real.** Aproximadamente 71% de los usuarios tienen exactamente 1 interacción registrada (features casi idénticas entre sí) — una concentración de valores casi-duplicados que no está presente en los datos sintéticos usados para el benchmark de escalabilidad, y que podría afectar de forma distinta la convergencia de cuML frente a Scikit-Learn (hipótesis no verificada en esta entrega).

### Reducción dimensional (UMAP) e interpretación de segmentos

Se proyectaron 50,000 usuarios (muestra) a 2D con UMAP (GPU), coloreados por cluster, para inspeccionar visualmente la separación entre segmentos.

| Cluster | % de la muestra | Dispersión (std) en el plano UMAP |
|---|---|---|
| Mayoritario (~76.4% de la muestra) | 76.4% | Alta (σx≈78.4, σy≈86.4) |
| Los otros 5 clusters | 23.6% combinado | Baja (σ entre 0.1 y 11.6) |

El cluster mayoritario —consistente con el segmento de usuarios de una sola interacción ya identificado en la Entrega 2 (71% de la base)— muestra una dispersión inesperadamente alta en el plano UMAP, a pesar de estar compuesto por usuarios con features casi idénticas. Esto es contraintuitivo si se espera que usuarios homogéneos formen un grupo compacto, pero es consistente con un artefacto conocido de UMAP: cuando una fracción muy grande de puntos son casi-duplicados en el espacio original, el grafo de vecinos que el algoritmo construye para esa masa de puntos queda mal condicionado, y UMAP tiende a esparcirlos de forma poco estable en el embedding 2D, sin que esto refleje heterogeneidad real en los datos originales. Los otros cinco clusters, compuestos por usuarios con mayor variabilidad genuina en su comportamiento, aparecen como grupos compactos y bien diferenciados — consistente con una segmentación significativa para esos segmentos.

*Nota metodológica:* las etiquetas numéricas de cluster asignadas en esta corrida (GPU, Colab) no son directamente comparables con las de la Entrega 2, ya que K-Means asigna las etiquetas de forma arbitraria en cada entrenamiento independiente. La comparación válida es a nivel de patrón (un segmento mayoritario de baja actividad vs. varios segmentos minoritarios diferenciados), no de número de cluster.

**Implicación para el sistema de recomendación:** el hecho de que el segmento mayoritario tenga una representación dispersa/inestable en el espacio reducido, mientras los segmentos minoritarios son compactos, es consistente con el hallazgo ya documentado en M3: el modelo de recomendación tiene más dificultad prediciendo para usuarios con patrones de interacción menos definidos (una sola interacción) que para usuarios con comportamiento más consistente — la geometría del embedding y el desempeño del modelo cuentan la misma historia desde dos ángulos distintos.

### Recomendación justificada

A diferencia de M1 (cuDF gana por un margen enorme) y M3 (NCF se beneficia claramente de GPU una vez resueltos los cuellos de botella de implementación), **para este pipeline específico de clustering, con la implementación actual, Scikit-Learn sobre CPU resulta más rápido en producción que cuML sobre GPU** — no porque K-Means en GPU sea inherentemente peor (el benchmark aislado confirma lo contrario), sino porque el overhead de transferencia del preprocesamiento adicional (estandarización) consume la ganancia que el algoritmo de clustering por sí solo sí obtiene en GPU.

Este resultado es valioso precisamente por no encajar con la narrativa de "GPU siempre gana" construida en M1 y M3: confirma que la decisión de acelerar por GPU debe evaluarse a nivel de **pipeline completo**, no solo del algoritmo central, y que mover únicamente el paso de cómputo más pesado a GPU sin considerar el resto de las operaciones circundantes puede no producir la mejora esperada.

### Trabajo futuro

- Perfilar por separado el tiempo de `log1p`, transferencia a `cudf`, y `StandardScaler` en el pipeline GPU, para confirmar cuál de los pasos adicionales explica la mayor parte de la sobrecarga.
- Evaluar si mantener el preprocesamiento en CPU (Polars/Scikit-Learn) y transferir a GPU únicamente para el paso de K-Means recupera la ventaja observada en el benchmark aislado.
- Repetir la comparación con `n_init` igualado explícitamente entre ambas implementaciones para aislar ese factor.
