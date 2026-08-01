## M1 — Escalabilidad fuerte y débil (Entrega 3)

### Análisis de resultados

**Escalabilidad fuerte.** Se midió el tiempo de carga y limpieza de `events.csv` completo variando el número de workers de Dask (1, 2, 4, 8), manteniendo el dataset fijo.

| Workers | Tiempo (s) | Speedup | Eficiencia |
|---|---|---|---|
| 1 | 7.02 | 1.00x | 1.000 |
| 2 | 4.92 | 1.43x | 0.714 |
| 4 | 4.29 | 1.64x | 0.409 |
| 8 | 2.93 | 2.40x | 0.300 |

Ajustando la Ley de Amdahl sobre estas cuatro configuraciones se obtiene una fracción secuencial estimada del **37.0%**, con un techo teórico de speedup de **2.70x**. Este resultado mejora sobre la fracción secuencial del 54.2% medida en la Entrega 2 bajo el mismo código — una posible explicación es la mayor consistencia en el número de particiones generadas (5, de forma estable) frente a la variabilidad observada anteriormente. En cualquier caso, ambas mediciones son coherentes entre sí en su conclusión de fondo: el pipeline de ETL tiene una fracción secuencial estructural relevante que limita el speedup alcanzable únicamente mediante más workers.

**Escalabilidad débil.** Se midió el tiempo de la misma operación, pero haciendo crecer el dataset proporcionalmente al número de workers (300,000 filas para 1 worker, hasta 2,400,000 filas para 8 workers).

| Workers | Filas | Tiempo (s) | Eficiencia débil |
|---|---|---|---|
| 1 | 300,000 | 0.72 | 1.000 |
| 2 | 600,000 | 1.57 | 0.457 |
| 4 | 1,200,000 | 2.11 | 0.339 |
| 8 | 2,400,000 | 2.98 | 0.240 |

En un sistema con escalabilidad débil ideal, el tiempo se mantendría aproximadamente constante al crecer datos y recursos en la misma proporción. Aquí no ocurre así: la eficiencia cae a 0.24 en la configuración más grande. Descomponiendo el tiempo total como una función lineal del número de workers, $T(N) = a + b \cdot N$, el ajuste por mínimos cuadrados da:

$$T(N) \approx 0.739 + 0.295 \cdot N \quad \text{(segundos)}$$

Esta descomposición separa el tiempo en un **componente fijo** (~0.74s, correspondiente al costo de inicialización del pipeline — construcción del grafo de tareas de Dask, arranque del clúster local — independiente del volumen de datos) y un **componente que crece linealmente con N** (~0.295s por unidad de carga añadida). Bajo escalabilidad débil ideal, este segundo componente debería mantenerse aproximadamente constante, ya que cada worker adicional procesa una porción de datos proporcional a su capacidad. Que en cambio crezca de forma lineal con N indica que el costo de coordinación entre workers escala con el número de workers, no solo con el volumen de datos — el mismo tipo de limitación estructural identificado en la Entrega 2 al analizar la granularidad de partición del ETL.

### Gestión de problemas

**Problema — El pipeline no logra escalabilidad débil real.** Aunque el sistema muestra un speedup razonable en el régimen de escalabilidad fuerte (2.40x con 8 workers sobre un dataset fijo), no mantiene el tiempo de respuesta constante cuando datos y recursos crecen juntos, que es el escenario más representativo de un pico de tráfico real (p. ej. Black Friday). La causa identificada es que el overhead de coordinación de Dask (construcción y programación del grafo de tareas) escala con el número de workers involucrados, no solo con el volumen de datos —de modo que agregar más workers en proporción al crecimiento de tráfico no garantiza, con la configuración actual, que el tiempo de respuesta se mantenga estable.

**Implicación para el diseño del sistema.** Este hallazgo es relevante para la simulación de picos de tráfico prevista en el resto de la Entrega 3: escalar horizontalmente el número de workers de Dask no es, por sí solo, una estrategia suficiente para sostener el tiempo de respuesta bajo carga creciente. Se documenta como limitación conocida del enfoque actual, y se identifica como candidato a explorar en la etapa de escalabilidad del sistema integrado (M4): evaluar si un tamaño de partición fijo y mayor (reduciendo el número de tareas coordinadas por worker) mitiga este efecto, en línea con el hallazgo de granularidad de partición ya documentado en la Entrega 2.
