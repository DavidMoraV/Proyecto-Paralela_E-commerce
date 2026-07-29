## Modelado (M3)

### Arquitectura del modelo baseline

El modelo baseline del sistema de recomendación implementa **filtrado colaborativo con feedback implícito mediante factorización matricial (ALS — Alternating Least Squares)**, siguiendo la formulación de Hu, Koren y Volinsky (2008), a través de la librería `Implicit`, que paraleliza el entrenamiento en múltiples hilos de CPU.

El modelo representa cada usuario y cada producto como un vector de características latentes (`factors=64` dimensiones). La predicción de afinidad entre un usuario *u* y un producto *i* se estima como el producto punto de sus vectores latentes respectivos. A diferencia del filtrado colaborativo clásico basado en calificaciones explícitas, este modelo no predice una calificación, sino un **score de preferencia relativo**, adecuado para el tipo de datos disponible en RetailRocket (vistas, adiciones al carrito y compras — nunca una calificación directa del usuario).

### Función de pérdida y señal de confianza

Siguiendo el marco de feedback implícito, cada interacción se traduce en un valor de **confianza** en lugar de una calificación:

```
peso_implicito = { 1  si event = view
                  { 3  si event = addtocart
                  { 5  si event = transaction
```

Esta ponderación (definida en M1, `transformar_eventos`) refleja que una compra es una señal de interés mucho más fuerte que una simple visualización. El modelo ALS minimiza una función de costo de mínimos cuadrados regularizada, ponderada por esta confianza, alternando entre resolver los vectores de usuario y los vectores de producto manteniendo el otro fijo en cada iteración — de ahí el nombre "Alternating" Least Squares. Este esquema es el que se paraleliza internamente entre hilos de BLAS, tal como se documentó en el análisis de rendimiento (M4).

### Configuración de entrenamiento

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `factors` (dimensión latente) | 64 | Punto de partida razonable para una primera validación; candidato a ajuste con Optuna en la Entrega 3 |
| `regularization` | 0.01 | Valor por defecto de la librería; evita sobreajuste dada la alta dispersión de la matriz |
| `iterations` | 15 | Suficiente para convergencia observada sin costo de entrenamiento excesivo (~100s sobre el dataset completo) |
| División train/test | Leave-one-out temporal | Para cada usuario con ≥2 interacciones, la más reciente se aparta como test; el resto se usa para entrenar |

**Matriz de interacciones usuario-producto:** 1,407,580 usuarios × 227,081 productos, con 1,930,255 interacciones no nulas (densidad de 0.0006%) — consistente con lo esperado para un catálogo de e-commerce de este tamaño, donde cada usuario interactúa solo con una fracción mínima del catálogo total.

### Métricas de evaluación

Se evaluó el modelo con tres métricas estándar de sistemas de recomendación basados en ranking, sobre 397,571 usuarios (97.9% del conjunto de test evaluable):

| Métrica | Definición | Resultado |
|---|---|---|
| **Hit Rate@10** | Fracción de usuarios donde el producto real aparece entre las 10 recomendaciones | **0.0266** |
| **MAP@10** | Precisión promedio, ponderada por la posición exacta del acierto dentro del top-10 | **0.0124** |
| **NDCG@10** | Ganancia acumulada descontada por posición (Järvelin & Kekäläinen, 2002) | **0.0157** |

Estos valores son el punto de partida esperado para un modelo baseline sin ajuste de hiperparámetros, evaluado contra un catálogo de más de 227,000 productos candidatos — el objetivo de esta entrega es establecer una línea base reproducible y medible, no maximizar la métrica todavía.

### Integración con M2 — rendimiento por segmento de usuario

Aprovechando los 6 clusters de usuario ya identificados en M2, se descompuso el Hit Rate@10 por segmento:

| Cluster (perfil, según M2) | Hit Rate@10 | Usuarios evaluados |
|---|---|---|
| Cluster 0 — Bajo/moderado | 0.0289 | 305,780 |
| Cluster 3 — Sobre el promedio | 0.0231 | 57,369 |
| Cluster 5 — Actividad intermedia | 0.0132 | 22,801 |
| Cluster 4 — Orientado a compra | 0.0066 | 10,383 |
| Cluster 2 — Muy activo | 0.0057 | 1,238 |
| Cluster 1 — Ocasional (1 interacción) | No evaluable | — (excluido por diseño, ver "Gestión de problemas") |

**Hallazgo relevante:** el cluster de usuarios más activos (Cluster 2) tiene el peor Hit Rate, no el mejor. La explicación más plausible es que estos usuarios interactúan con un catálogo mucho más amplio y diverso (~66 productos distintos en promedio), lo que hace su siguiente interacción intrínsecamente más difícil de predecir que la de un usuario con intereses más concentrados. Más interacciones no implica una recomendación más fácil — de hecho, en este caso ocurre lo contrario.

### Análisis de resultados: magnitud de los scores

Los scores absolutos generados por el modelo (usados para ordenar las recomendaciones) son del orden de 10⁻⁹–10⁻¹⁰, consecuencia de la alta dispersión de la matriz combinada con la regularización L2, que empuja los vectores latentes hacia magnitudes pequeñas cuando hay pocas interacciones para restringirlos. Esto no afecta la validez del modelo — el score de ALS es un producto punto entre vectores latentes, no una probabilidad, y lo relevante para la recomendación es el **orden relativo** entre productos candidatos, no la magnitud absoluta del score.

### Gestión de problemas

**Cold start de usuarios de una sola interacción.** El Cluster 1 (~71% de la base de usuarios) tiene una única interacción registrada, por lo que la estrategia leave-one-out no puede evaluarlos (no queda nada en train tras apartar su único evento). Esta es una limitación conocida del filtrado colaborativo puro; un modelo híbrido con componente de popularidad o basado en contenido sería necesario para dar cobertura a este segmento, y se propone como trabajo futuro para la Entrega 3.

**Dispersión extrema de la matriz.** Con 0.0006% de densidad, los factores latentes de usuarios/productos con pocas interacciones están débilmente restringidos por los datos. Se documenta como limitación conocida del baseline; el ajuste de `regularization` y `factors` vía búsqueda de hiperparámetros (Optuna, Entrega 3) es la vía planeada para mitigarlo.

### Próximos pasos (Entrega 3)

- Comparar este baseline ALS contra un modelo de embeddings/Two-Tower (PyTorch), evaluado con las mismas tres métricas sobre el mismo conjunto de test.
- Ajuste de hiperparámetros (`factors`, `regularization`, `iterations`) con Optuna.
- Evaluar una estrategia de respaldo (popularidad o basada en contenido) para el segmento de usuarios de una sola interacción.
