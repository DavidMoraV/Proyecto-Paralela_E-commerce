## M1  Comparativa Final: Pandas vs Polars vs Dask vs cuDF (Entrega 3)

### Resultados

Se midió el tiempo de carga, limpieza y transformación de `events.csv` completo con las cuatro herramientas evaluadas en el proyecto, en un entorno con GPU (Google Colab, T4).

| Herramienta | Carga | Limpieza | Transformación | Total | Speedup | Recursos usados | Eficiencia |
|---|---|---|---|---|---|---|---|
| Pandas (baseline) | 1.78 s | 1.56 s | 0.46 s | 3.80 s | 1.00x | 1 núcleo | 1.000 |
| Polars | 0.44 s | 0.81 s | 0.17 s | 1.42 s | 2.67x | 2 núcleos | 1.335 |
| Dask | 1.59 s | 2.51 s | 0.64 s | 4.75 s | 0.80x | 4 workers | 0.200 |
| **cuDF (GPU)** | 0.07 s | 0.03 s | 0.05 s | **0.15 s** | **25.55x** | 1 GPU | 25.55 |


### Nota metodológica: el desglose por etapa de Dask no refleja dónde ocurre el costo real

Antes de interpretar los números de Dask por etapa, es necesario señalar una limitación de la medición: Dask construye sus operaciones de forma perezosa (*lazy evaluation*) — las funciones de carga y limpieza solo arman el plan de ejecución, sin materializar ningún dato. Solo al llamar `.compute()`, al final de la etapa de transformación, Dask ejecuta **todo el plan acumulado de las tres etapas de una sola vez**. Esto explica por qué `carga` (0.005s) y `limpieza` (0.009s) aparecen artificialmente rápidas — no hicieron trabajo real todavía — mientras que `transformación` (4.66s) absorbe el costo computacional completo de las tres etapas juntas, no solo el suyo.

A diferencia de Dask, la implementación de Polars sí materializa (`.collect()`) al final de la etapa de carga, por lo que su desglose por etapa es una medición justa. El **tiempo total** de Dask (4.68s) sigue siendo una comparación válida frente a las demás herramientas, pero su desglose por etapa individual no debe usarse como evidencia de que "la transformación es el paso lento" en Dask — es un artefacto de la evaluación perezosa, no una propiedad real del cómputo. Se corrige esta medición en la siguiente iteración (ver "Gestión de problemas").

### Análisis de resultados

**cuDF (GPU) domina de forma contundente**, con un speedup de **25.55x** sobre Pandas — muy por encima de Polars (2.67x) y claramente superior a Dask en esta configuración (0.80x, es decir, más lento que el baseline secuencial). Esto contrasta con la cautela inicial planteada antes de correr el experimento: la hipótesis de que "el volumen de datos podría no ser suficiente para amortizar el overhead de transferencia CPU→GPU" no se sostiene con la evidencia — para 2.7M de eventos, la aceleración GPU es, con diferencia, la más efectiva de las cuatro opciones evaluadas.

**Dask resulta la opción menos favorable en este entorno específico**, incluso por debajo del baseline secuencial (Pandas). Con la medición ya corregida (ver "Gestión de problemas"), se confirma que la etapa de limpieza (deduplicación) es el verdadero cuello de botella de Dask, no la transformación como sugería la medición original defectuosa. Esto es consistente con dos hallazgos ya documentados en entregas anteriores: (1) Dask requiere mayor volumen de datos y mejor calibración de particiones para amortizar su overhead de coordinación (Entrega 2), y (2) en este entorno de Colab, la eficiencia de 0.200 (con 4 workers configurados sobre lo que probablemente son solo 2 núcleos de CPU disponibles, a juzgar por el conteo de recursos de Polars) sugiere sobresuscripción de hilos — pedir más paralelismo del que el hardware puede sostener, el mismo patrón de oversubscription ya identificado con OpenBLAS durante el entrenamiento de ALS.

**Polars sigue siendo una opción sólida y de bajo costo operativo**: 2.67x de speedup sin requerir GPU ni configuración de clúster, la opción más simple de desplegar de las cuatro.

### Recomendación

| Escenario | Herramienta recomendada | Justificación |
|---|---|---|
| Producción con GPU disponible de forma consistente | **cuDF** | Speedup de 24.85x justifica ampliamente la complejidad adicional de la infraestructura GPU |
| Desarrollo/entornos sin GPU garantizada | **Polars** | 3.25x de speedup, sin dependencias de hardware especializado, más simple de mantener |
| Distribución en clúster multi-nodo real (no local) | Dask, con ajuste de particiones | Los resultados actuales no favorecen a Dask, pero su ventaja real aparece al distribuir entre **nodos** físicos distintos, no solo hilos en una máquina — escenario no cubierto por esta medición |

Para el pipeline de este proyecto específicamente, dado que ya se cuenta con acceso a GPU a través de Colab para el entrenamiento del modelo (M3), extender su uso a la etapa de ETL con cuDF es la recomendación técnica más defendible: la ganancia de rendimiento (25.55x) es sustancialmente mayor que cualquier otra alternativa evaluada, y la infraestructura GPU ya es parte del flujo de trabajo del proyecto.

### Gestión de problemas: medición por etapa de Dask corregida, de antes/después

**Detección.** En la primera corrida, el desglose por etapa de Dask mostraba `carga: 0.005s`, `limpieza: 0.009s`, `transformación: 4.66s` — una distribución no creíble, con el 99.7% del tiempo concentrado en la última etapa.

**Causa raíz.** Dask evalúa de forma perezosa (*lazy*): las funciones de carga y limpieza solo construían el plan de ejecución sin materializar datos. Solo al llamar `.compute()`, al final de la transformación, Dask ejecutaba **todo el plan acumulado de las tres etapas de una sola vez**, atribuyendo su costo completo a la última etapa medida.

**Solución aplicada.** Se reemplazó `.compute()` por `.persist()` al final de `dask_carga` y `dask_limpieza`. A diferencia de `.compute()` (que convierte a pandas), `.persist()` fuerza el cómputo de esa etapa específica de inmediato pero mantiene el resultado como una colección Dask — así la siguiente etapa construye sobre un resultado ya materializado, no sobre el grafo completo desde cero.

**Verificación cuantitativa (antes/después):**

| Etapa | Antes (defectuoso) | Después (corregido) |
|---|---|---|
| Carga | 0.005 s | 1.59 s |
| Limpieza | 0.009 s | 2.51 s |
| Transformación | 4.66 s | 0.64 s |
| **Total** | 4.68 s | 4.75 s |

El tiempo **total** apenas cambió (4.68s → 4.75s, dentro del margen de variabilidad esperado), confirmando que la comparación de totales entre herramientas ya era válida desde la primera corrida — lo que estaba mal era exclusivamente la atribución del tiempo entre etapas. Con la corrección, la etapa dominante para Dask resulta ser **limpieza** (2.51s, 53% del total), consistente con el hallazgo ya documentado en la Entrega 2 de que la deduplicación es la operación más costosa del pipeline de Dask, al requerir coordinación (shuffle) entre particiones la corrección no solo arregla un bug, sino que además refuerza con nueva evidencia una conclusión ya establecida en el proyecto.
