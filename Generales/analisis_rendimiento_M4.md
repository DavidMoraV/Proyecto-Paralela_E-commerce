## Análisis de Rendimiento (M4)

### Configuración experimental

Todas las mediciones siguen el mismo protocolo: una repetición de calentamiento (warm-up) descartada, seguida de 2-3 repeticiones que sí se promedian, reportando media, desviación estándar, mínimo y máximo para cada configuración. Las mediciones se corrieron en una máquina de uso general (no un clúster dedicado ni un entorno con recursos reservados), por lo que se documenta como limitación conocida la posible interferencia de procesos del sistema operativo (ver "Gestión de problemas").

Se midieron tres etapas del pipeline — preprocesamiento (M1), clustering de usuarios (M2) y entrenamiento del modelo de recomendación (M3) — variando en cada una el grado de paralelismo disponible (workers de Dask para el ETL, hilos de BLAS/OpenMP para clustering y entrenamiento), sobre el dataset completo de RetailRocket (2,755,641 eventos, 1,407,580 usuarios).

### Resultados por etapa

| Etapa | Tiempo secuencial | Mejor tiempo medido | Speedup máximo | Fracción secuencial (Amdahl) | Techo teórico de speedup |
|---|---|---|---|---|---|
| ETL (M1) | 5.37 s | 2.83 s (8 workers) | 1.90x | 54.2% | 1.85x |
| Clustering (M2) | 11.07 s | 7.31 s (8 hilos) | 1.51x | 60.4% | 1.65x |
| Entrenamiento ALS (M3) | 106.91 s | 97.93 s (8 hilos BLAS) | 1.09x | 89.2% | 1.12x |

La fracción secuencial se estimó ajustando la Ley de Amdahl (S(N) = 1 / (S + (1-S)/N)) por mínimos cuadrados sobre las cuatro configuraciones medidas (1, 2, 4 y 8 unidades de paralelismo) de cada etapa, en vez de estimarla a partir de un solo punto — las estimaciones individuales por punto variaron considerablemente entre sí (ver "Gestión de problemas"), lo que hace preferible el ajuste conjunto.

El resultado más relevante de esta tabla es que **la etapa que consume más del 90% del tiempo total del pipeline (el entrenamiento) es también la que menos margen tiene para mejorar mediante paralelismo CPU** (89.2% de fracción secuencial, techo de apenas 1.12x). Las otras dos etapas sí escalan de forma razonable con más hilos/workers, pero su peso conjunto es marginal frente al entrenamiento.

### Hallazgo adicional: la granularidad de partición importa más que la cantidad

Se probó la hipótesis de que la baja eficiencia del ETL se debía a un número insuficiente de particiones (5, con el `blocksize` original de 16MB). Al reducir el `blocksize` a 4MB se generaron 23 particiones, pero el resultado no confirmó la hipótesis de forma simple:

| Configuración | Tiempo secuencial | Mejor configuración | Comportamiento a partir de 4 workers |
|---|---|---|---|
| 16MB (5 particiones) | 5.37 s | 8 workers (1.90x) | Sigue mejorando hasta 8 workers |
| 4MB (23 particiones) | 3.81 s | **2 workers (1.55x)** | **Empeora** (2.79s → 2.83s) |

Con más particiones, el tiempo secuencial de base mejoró (menos memoria por partición), y el punto óptimo de paralelismo se alcanzó antes (en 2 workers), pero agregar más workers después de ese punto degradó el rendimiento en vez de mejorarlo. Esto indica que el cuello de botella no es simplemente "pocas particiones", sino un equilibrio entre el tamaño de cada partición y el número de workers: particiones demasiado pequeñas generan overhead de coordinación que supera el trabajo real que hay que repartir.

### Cuellos de botella identificados

1. **El entrenamiento del modelo domina el tiempo total del pipeline (>90%) y es la etapa con menor margen de mejora vía paralelismo CPU** (Amdahl: 89.2% secuencial, techo de 1.12x). Es el candidato principal para optimización en la Entrega 3, probablemente mediante aceleración GPU (cuML/PyTorch) en vez de más hilos CPU.

2. **La granularidad de partición del ETL no está calibrada**: ni "pocas particiones grandes" ni "muchas particiones pequeñas" es óptimo por sí solo — existe un punto intermedio (evidenciado por el cambio de óptimo entre 8 workers/5 particiones y 2 workers/23 particiones) que no se ha determinado con precisión y debería explorarse con más configuraciones intermedias.

3. **Rendimientos marcadamente decrecientes en las tres etapas**: la eficiencia cae por debajo de 0.25 en las tres etapas al llegar a 8 unidades de paralelismo (ETL: 0.237, Clustering: 0.189, Entrenamiento: 0.136). El pipeline, en su forma actual, no aprovecha bien más de 2-4 unidades de paralelismo simultáneas — agregar más recursos de cómputo más allá de ese punto tiene un retorno marginal decreciente y, en el caso del ETL con particiones pequeñas, incluso negativo.

### Gestión de problemas

**Problema 1 — Variabilidad entre repeticiones.** Se observó dispersión no despreciable entre repeticiones de una misma configuración (por ejemplo, el ETL a 8 workers/5 particiones varió entre 2.44s y 3.36s). Esto se refleja en que las estimaciones de la fracción secuencial de Amdahl calculadas con un solo punto (N=8) difieren de las calculadas con otro punto (N=2) en más de 25 puntos porcentuales para algunas etapas. Se mitiga reportando el ajuste conjunto sobre las cuatro configuraciones en vez de un solo punto, y se recomienda aumentar el número de repeticiones (de 2-3 a 6-8) para la versión final de estas mediciones en la Entrega 3.

**Problema 2 — `cProfile` no es una herramienta adecuada para perfilar este entrenamiento.** Se intentó perfilar `modelo.entrenar()` con `cProfile`, pero los resultados no fueron interpretables: las funciones reales de ALS (`fit`, `entrenar`) aparecieron con tiempos acumulados cercanos a cero, mientras que funciones de espera de hilos (`select`, `wait`, `acquire`) dominaron el reporte con decenas de segundos. La causa es que `cProfile` solo instrumenta el hilo principal de Python; como Implicit libera el GIL y ejecuta el cómputo real en hilos nativos de BLAS, el hilo principal solo registra que está esperando a que esos hilos terminen. Esta limitación metodológica se documenta en vez de reportar resultados de profiling engañosos; para obtener una atribución de tiempo confiable dentro de código nativo multihilo haría falta una herramienta de profiling a nivel de sistema operativo (ej. `py-spy`, Intel VTune), fuera del alcance de esta entrega. El experimento de variar hilos de BLAS (sección de resultados) se considera el enfoque válido para este caso, ya que mide el efecto real sobre el tiempo de pared sin depender de la instrumentación interna de Python.

**Problema 3 — Entorno de medición no aislado.** Las mediciones se corrieron en una máquina de uso general, no en un clúster dedicado (como Kabré, CeNAT) ni en un contenedor con recursos reservados. Se documenta como limitación conocida; de tener acceso a un entorno dedicado, se recomienda repetir estas mediciones ahí para la Entrega 3, donde además se plantean escenarios de escalabilidad fuerte y débil bajo picos de tráfico simulados.

### Conclusión y siguientes pasos

La evidencia recolectada en esta entrega apunta a una conclusión clara: seguir agregando paralelismo CPU (más workers de Dask, más hilos de BLAS) tiene rendimientos decrecientes en las tres etapas del pipeline, y en particular no representa una vía de mejora significativa para el entrenamiento del modelo, que es la etapa dominante en tiempo. Esto justifica, con evidencia empírica propia y no solo con intención de diseño, la decisión ya tomada en la justificación técnica del proyecto de explorar aceleración GPU (RAPIDS cuML / PyTorch) para la Entrega 3, en vez de continuar optimizando exclusivamente por la vía de más paralelismo CPU.
