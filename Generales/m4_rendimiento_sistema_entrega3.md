## M4 — Análisis de Rendimiento del Sistema Integrado (Entrega 3)

### Análisis de resultados

**Pipeline completo, de punta a punta.** Se midió el sistema completo — ingesta y limpieza (M1), clustering (M2), entrenamiento (M3) e inferencia batch (M3) — en una sola ejecución continua, en vez de medir cada etapa aislada como en la Entrega 2.

| Etapa | Tiempo | % del total |
|---|---|---|
| M1 — Ingesta y limpieza | 3.45 s | 2.5% |
| M2 — Clustering | 12.05 s | 8.8% |
| M3 — Entrenamiento | 120.43 s | 87.9% |
| M3 — Inferencia batch (1,000 usuarios) | 1.03 s | 0.8% |
| **Total** | **136.96 s** | 100% |

El entrenamiento domina el sistema completo con el **87.9%** del tiempo total, resultado consistente con la fracción secuencial del 89.2% obtenida mediante el ajuste de la Ley de Amdahl sobre esta misma etapa en la Entrega 2 (medida de forma aislada, en una sesión distinta). La coincidencia entre ambas mediciones —una mirando el sistema completo, otra mirando solo el entrenamiento— refuerza la conclusión ya adelantada: cualquier esfuerzo de optimización de rendimiento en este sistema debe priorizar el entrenamiento del modelo por encima de las demás etapas, ya que el resto del pipeline (ingesta, clustering, inferencia) juntas representan apenas el 12.1% del tiempo total.

**Latencia de inferencia según tamaño de lote.** Se midió la latencia y el throughput de generar recomendaciones para lotes de usuarios de tamaño creciente (1 a 10,000), simulando el escenario de servir recomendaciones en producción bajo distintos niveles de carga concurrente.

| Usuarios por lote | Latencia total | Latencia por usuario | Throughput |
|---|---|---|---|
| 1 | 6.4 ms | 6.40 ms | 156 usuarios/s |
| 10 | 33.4 ms | 3.34 ms | 299 usuarios/s |
| 100 | 151.4 ms | 1.51 ms | 660 usuarios/s |
| 1,000 | 1.11 s | 1.11 ms | 899 usuarios/s |
| 10,000 | 9.47 s | 0.95 ms | 1,056 usuarios/s |

El costo por usuario individual mejora consistentemente al agrupar más solicitudes en un mismo lote (de 6.40 ms a 0.95 ms, una mejora de 6.7×), lo cual es el comportamiento esperado: el overhead fijo de cada llamada se amortiza entre más usuarios. Sin embargo, el throughput agregado presenta **rendimientos marcadamente decrecientes**: al pasar de 1,000 a 10,000 usuarios (un incremento de 10×), el throughput solo mejora un 17% (de 899 a 1,056 usuarios/s), mientras que la latencia total del lote se dispara de 1.11 a 9.47 segundos.

Esto tiene una implicación directa para el escenario de picos de tráfico (simulación tipo Black Friday) que contempla el proyecto: agrupar todas las solicitudes concurrentes en un único lote grande no es una estrategia viable, ya que cualquier usuario cuya solicitud caiga en un lote de 10,000 esperaría hasta 9.5 segundos por su recomendación — muy por encima de lo aceptable para una experiencia en tiempo real. Los datos sugieren que un tamaño de lote intermedio (en el rango de 100 a 1,000 usuarios) ofrece el mejor equilibrio entre throughput agregado y latencia de cola, y es el rango recomendado para el diseño de la capa de servicio del sistema.

**Balance de carga y costos de comunicación.** Se midió, sobre un clúster local de Dask distributed con 4 workers, cómo se repartió el trabajo durante una operación con shuffle real entre particiones (deduplicación de eventos).

| Métrica | Valor |
|---|---|
| Workers con trabajo asignado | 4 de 4 |
| Coeficiente de variación del tiempo ocupado | 0.115 |
| Tiempo total de transferencia entre workers | 0.494 s |

Con el dataset completo, los 4 workers recibieron trabajo (a diferencia de pruebas preliminares con muestras pequeñas, donde particiones insuficientes dejaban workers ociosos — el mismo fenómeno de granularidad de partición ya documentado en la Entrega 2). El coeficiente de variación de 0.115 indica un reparto de carga razonablemente parejo entre los 4 workers. El tiempo de comunicación (0.494 s) representa aproximadamente un 3.7% de overhead sobre el tiempo agregado de cómputo de los workers (~13.2 s), cuantificando por primera vez en el proyecto el costo real de coordinación distribuida — un dato que hasta ahora solo se había inferido indirectamente a partir de la caída de eficiencia al agregar más workers.

**Mejoras aplicadas, con impacto cuantitativo.**

| Mejora | Etapa afectada | Impacto medido |
|---|---|---|
| Detección automática de separador/decimal en la ingesta | M1 — Ingesta | Evitó perder 13.0% de eventos por falsos duplicados (136,164 filas) |
| Evaluación muestreada (1 positivo + 99 negativos) en vez de catálogo completo | M3 — Evaluación | Evaluación de ALS/NCF tratable en minutos en vez de horas |
| Indexación vectorizada de tensores en vez de `DataLoader` | M3 — Entrenamiento NCF | Eliminó el cuello de botella real detectado durante el entrenamiento en Colab (ver Gestión de problemas) |

### Conclusión

La evidencia recolectada en esta sección converge hacia una misma conclusión desde tres ángulos distintos: el análisis por etapa (Entrega 2), el pipeline de punta a punta (esta sección) y el balance de carga distribuido apuntan todos a que el entrenamiento del modelo es el cuello de botella dominante del sistema, y que el paralelismo CPU adicional tiene rendimientos decrecientes tanto en cómputo como en coordinación entre workers. El hallazgo de latencia de inferencia añade una dimensión nueva y práctica: incluso si el entrenamiento se optimizara significativamente (por ejemplo, con aceleración GPU), el diseño de la capa de servicio en producción requiere una política explícita de tamaño de lote — no simplemente "procesar todo junto" — para sostener tiempos de respuesta aceptables bajo picos de tráfico reales.
