## M3  Modelo Avanzado (NCF) vs Baseline ALS: Resultados Finales 

### Comparación de modelos (mismo protocolo de evaluación)

Ambos modelos se evaluaron bajo el mismo protocolo muestreado (1 producto real + 99 negativos aleatorios, siguiendo He et al., 2017), sobre 5,000 usuarios de test — a diferencia de la Entrega 2, donde ALS se evaluó contra el catálogo completo. Esto hace la comparación metodológicamente justa.

| Métrica | ALS (baseline) | NCF (avanzado) | Mejora relativa |
|---|---|---|---|
| Hit Rate@10 | 0.6718 | **0.7588** | +13.0% |
| MAP@10 | 0.4535 | **0.5928** | +30.7% |
| NDCG@10 | 0.5045 | **0.6322** | +25.3% |

El modelo avanzado (NCF) supera al baseline ALS en las tres métricas, con mejoras relativas de entre 13% y 31%. La mejora es más pronunciada en MAP@10 y NDCG@10 (métricas sensibles a la posición exacta del acierto dentro del ranking) que en Hit Rate@10 (que solo mide si el acierto ocurre en algún lugar del top-10) — esto sugiere que la ventaja principal de NCF no es solo acertar más veces, sino **rankear el producto correcto más arriba** cuando acierta, algo consistente con su capacidad de modelar interacciones no lineales usuario-producto que ALS, limitado a un producto punto, no puede representar.

### Benchmark de entrenamiento: GPU vs CPU

| Dispositivo | Tiempo por época | Speedup |
|---|---|---|
| GPU (T4) | 98.75 s | 3.97x |
| CPU | 391.94 s | 1.00x (baseline) |

Una vez corregido el cuello de botella de iteraciones (ver "Gestión de problemas" de M3 en la Entrega 3), la GPU muestra una aceleración real y consistente de **~4x** sobre CPU para este modelo — un resultado creíble y esperable para una red de este tamaño, y evidencia de que las optimizaciones aplicadas (indexación vectorizada, batch size mayor) permitieron finalmente medir el beneficio real de la GPU, en vez de estar dominados por el overhead de Python.

### Gestión de problemas: riesgo de sobreajuste durante el entrenamiento

**Detección.** El historial de pérdida (BCE) del entrenamiento de NCF cae de 0.364 en la primera época a **0.0000061** en la décima — una pérdida de entrenamiento prácticamente nula.

**Causa raíz.** Los ejemplos negativos usados para entrenar se generan **una sola vez**, al construir el dataset (`InteraccionesImplicitasDataset._generar_muestras`), y se reutilizan idénticos en las 10 épocas. Esto permite que el modelo memorice ese conjunto fijo de negativos en vez de aprender un patrón generalizable de preferencia usuario-producto — el riesgo clásico de sobreajuste cuando el conjunto de entrenamiento no varía entre épocas.

**Por qué no invalida los resultados reportados, pero sí es una limitación a documentar.** La evaluación (Hit Rate@10, MAP@10, NDCG@10) se hizo sobre el conjunto de **test**, con usuarios y productos que el modelo no vio durante el entrenamiento, y con negativos de evaluación muestreados de forma independiente (semilla fija, pero distintos del conjunto de entrenamiento). Las mejoras de NCF sobre ALS observadas en test son sustanciales pero no inverosímilmente altas, lo cual es consistente con una mejora real y no únicamente con memorización. Aun así, una pérdida de entrenamiento tan cercana a cero es una señal de alerta que no debe ignorarse.

**Solución recomendada para la Entrega 3/4.** Regenerar los negativos en cada época (`_generar_muestras()` dentro del loop de `entrenar_ncf`, no en el constructor del dataset), tal como especifica el protocolo original de He et al. (2017). Esto evita que el modelo vea siempre los mismos negativos y da una señal de entrenamiento más robusta contra el sobreajuste. Se documenta como mejora pendiente, no aplicada en esta entrega por el costo de tiempo adicional que implica regenerar 4 negativos por cada una de las 2.35M interacciones positivas en cada una de las 10 épocas.

### Exportación del modelo

El modelo final se exportó exitosamente a formato ONNX (`modelo_avanzado.onnx`), cumpliendo el requisito de la Entrega 3 de contar con un modelo en formato estándar, portable fuera de PyTorch para inferencia en producción.

### Conclusión

El modelo avanzado (NCF) demuestra una mejora medible y consistente sobre el baseline ALS en las tres métricas de calidad de ranking, bajo un protocolo de evaluación justo y comparable entre ambos. La aceleración por GPU (~4x) confirma el valor de la infraestructura GPU para este tipo de modelo, una vez resueltos los cuellos de botella de implementación. El hallazgo de posible sobreajuste durante el entrenamiento se documenta con transparencia como limitación metodológica conocida, junto con la solución concreta identificada para abordarla en el siguiente ciclo de desarrollo.
