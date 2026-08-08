# Sistema de Recomendación Paralelo para E-Commerce

Proyecto del curso **Computación Paralela y Distribuida** — LEAD University
Prof. Johansell Villalobos Cubillo

Sistema de recomendación de productos construido sobre el dataset [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (2,755,641 eventos, 1,407,580 usuarios), con énfasis en demostrar y **medir** el uso de computación paralela y distribuida en cada etapa del pipeline: ingesta, EDA, modelado, análisis de rendimiento y despliegue.

**Estado actual: Entrega 3 — Proyecto Final Completo.**
Informes completos en [`Generales/`](Generales/) (ver índice de documentación al final de este README).

## Equipo

| Integrante | Módulo(s) |
|---|---|
| David Mora | M1, M3, M4, M5 |
| Ana María Ramírez | M2 (Entrega 1-2; Entrega 3 desarrollada como refuerzo del equipo) |
| María José, Siloé Campos, Julio | Coordinación y documentación general |

## Arquitectura del sistema

```
Datos crudos (RetailRocket)
        │
        ▼
  M1 · ETL (Polars + Dask, comparado contra cuDF/GPU)     → datos/*.parquet
        │
        ▼
  M2 · EDA y segmentación (K-Means CPU/Scikit-Learn        → datos/user_segments.parquet
        vs GPU/cuML + UMAP)
        │
        ▼
  M3 · Modelo de recomendación                             → modelos/modelo_avanzado.onnx
        (ALS baseline + NCF avanzado, comparados
        bajo protocolo de evaluación común)
        │
        ├──► M4 · Análisis de rendimiento
        │      (speedup por etapa, Amdahl, balance de
        │      carga, latencia de inferencia, sistema
        │      integrado de punta a punta)
        │
        ▼
  M5 · Dashboard (Dash, dockerizado) + prueba de carga
        (simulación de pico de tráfico tipo Black Friday)
```

## Resultados clave

### Entrega 2 (validación inicial, dataset completo)

| Métrica | Valor |
|---|---|
| Eventos procesados | 2,755,641 |
| Usuarios únicos | 1,407,580 |
| Speedup ETL (Polars vs Pandas) | 3.02x |
| Hit Rate@10 del modelo baseline (ALS) | 0.0266 (catálogo completo) |
| Speedup máx. entrenamiento ALS | 1.09x (fracción secuencial 89.2%, Ley de Amdahl) |

### Entrega 3 (sistema final, GPU, modelo avanzado)

| Área | Resultado |
|---|---|
| **M1 — cuDF vs Pandas/Polars/Dask** | cuDF gana por **25.55x** — la aceleración GPU más efectiva del proyecto |
| **M1 — Escalabilidad débil del ETL** | No ideal: overhead de coordinación crece con el número de workers, no solo con los datos |
| **M2 — K-Means CPU vs GPU (aislado)** | GPU gana desde 100K filas (1.4x–1.6x) |
| **M2 — Clustering en producción (con preprocesamiento)** | CPU gana (GPU 0.86x) — el overhead de transferencia del preprocesamiento revierte la ventaja del algoritmo aislado |
| **M3 — NCF vs ALS (mismo protocolo)** | NCF supera a ALS: +13% Hit Rate@10, +31% MAP@10, +25% NDCG@10 |
| **M3 — GPU vs CPU (entrenamiento NCF)** | 3.97x, una vez corregidos 2 cuellos de botella de implementación |
| **M4 — Pipeline completo, de punta a punta** | Entrenamiento domina con 87.9% del tiempo total del sistema |
| **M4 — Latencia de inferencia** | De 6.4ms (1 usuario) a 9.47s (10,000 usuarios en un solo lote) |
| **M5 — Prueba de carga del dashboard** | Throughput estancado (~27-31 req/s) por el GIL de Python — requiere paralelismo por procesos, no por hilos, en producción |

Detalle completo de metodología, hallazgos y gestión de problemas en los informes de `Generales/` (ver índice abajo).

## Estructura del repositorio

```
.
├── Datasets/                          # Datos crudos de RetailRocket (NO incluido — ver "Cómo obtener los datos")
├── Generales/                         # Informes y documentos del curso
│   ├── Entrega1_Sistema_de_Recomendacion_Paralelo.pdf
│   ├── Entrega2_Sistema_de_Recomendacion_Paralelo.pdf   # Informe IEEE completo E2
│   ├── modelado_M3.md                                    # Modelado del baseline ALS (E2)
│   ├── entrega_M3_para_M4.md                              # Traspaso M3→M4 (E2)
│   ├── analisis_rendimiento_M4.md                         # Rendimiento por etapa (E2)
│   ├── gestion_de_problemas_M5.md                         # Integración del dashboard (E2)
│   ├── m1_escalabilidad_entrega3.md                       # Escalabilidad fuerte/débil (E3)
│   ├── m1_comparativa_final_entrega3.md                   # Pandas/Polars/Dask/cuDF (E3)
│   ├── m2_clustering_gpu_entrega3.md                      # K-Means CPU vs GPU + UMAP (E3)
│   ├── m3_resultados_finales_entrega3.md                  # NCF vs ALS (E3)
│   ├── m4_rendimiento_sistema_entrega3.md                 # Sistema integrado (E3)
│   └── m5_prueba_carga_entrega3.md                        # Prueba de carga (E3)
└── Codigos/
    ├── src/                            # Módulos Python reutilizables
    │   ├── pipeline_datos.py               # M1 — ingesta, limpieza, transformación, exportación
    │   ├── analisis_eda.py                 # M2 — features de usuario, clustering CPU
    │   ├── analisis_eda_gpu.py             # M2 — clustering GPU (cuML) + UMAP (E3)
    │   ├── modelo_recomendacion.py         # M3 — matriz de interacciones, ALS, evaluación
    │   ├── modelo_avanzado.py              # M3 — NCF (NeuMF), entrenamiento, exportación ONNX (E3)
    │   ├── benchmarks.py                   # M4 — speedup, Amdahl, profiling, escalabilidad
    │   └── analisis_sistema.py             # M4 — pipeline integrado, balance de carga (E3)
    ├── notebooks/                      # Notebooks que corren LOCAL (CPU, sin GPU)
    │   ├── PipelineM1.ipynb
    │   ├── M2_Analisis_EDA.ipynb
    │   ├── M3_Modelo_Recomendacion.ipynb
    │   ├── M4_Analisis_Rendimiento.ipynb
    │   ├── M1_Escalabilidad.ipynb                          # (E3)
    │   └── M4_Rendimiento_Sistema.ipynb                     # (E3)
    ├── colab/                          # Notebooks que requieren GPU — correr en Google Colab
    │   ├── M1_cuDF_vs_Polars_Dask.ipynb                     # (E3)
    │   ├── M2_Clustering_GPU_vs_CPU.ipynb                   # (E3)
    │   └── M3_Modelo_Avanzado_GPU.ipynb                     # (E3)
    ├── dashboard/
    │   ├── app.py                          # Dashboard interactivo (Dash), 5 pestañas
    │   ├── generar_dashboard_html.py       # Versión HTML estática, sin servidor
    │   ├── prueba_carga.py                 # Simulación de tráfico concurrente (E3)
    │   ├── Dockerfile                      # (E3)
    │   └── requirements.txt
    ├── modelos/                        # Modelos exportados
    │   └── modelo_avanzado.onnx            # NCF exportado a ONNX (E3)
    ├── datos/                          # Parquet procesados (NO incluido — se regenera con M1/M2)
    ├── resultados/                     # Evidencia de M1-M4 (CSV/PNG/JSON, SÍ incluido)
    │   └── entrega3/                       # Resultados nuevos de Entrega 3, separados de E2
    ├── docker-compose.yml              # (E3)
    ├── Makefile
    └── requirements.txt
```

## Cómo obtener los datos

El dataset no se incluye en el repositorio. Para reproducir el proyecto:

1. Descarga desde [Kaggle — RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
2. Coloca los 4 archivos (`events.csv`, `category_tree.csv`, `item_properties_part1.csv`/`.xlsx`, `item_properties_part2.csv`/`.xlsx`) dentro de `Datasets/`.

> **Nota:** el pipeline detecta y corrige automáticamente archivos que hayan pasado por Excel (separador `;`, decimales con `,`, timestamps en notación científica). Ver `Generales/gestion_de_problemas_M5.md` y `m1_escalabilidad_entrega3.md` para el detalle de estos casos reales encontrados durante el desarrollo.

## Instalación

```bash
pip install -r Codigos/requirements.txt
```

## Ejecución — parte local (CPU, sin GPU)

Desde `Codigos/`, en orden:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/PipelineM1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M2_Analisis_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M3_Modelo_Recomendacion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M4_Analisis_Rendimiento.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M1_Escalabilidad.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M4_Rendimiento_Sistema.ipynb
```

O con `make all` (Linux/Mac/Git Bash; en Windows sin `make`, correr los comandos manualmente).

## Ejecución — parte GPU (Google Colab)

Los tres notebooks de `Codigos/colab/` requieren GPU y **no se pueden correr localmente** en un entorno sin CUDA. Cada uno:

1. Clona este repositorio (`!git clone ...`) para traer los módulos de `src/`.
2. Pide subir los datos necesarios (`events.csv` o `eventos_limpios.zip`, según el notebook).
3. Corre el experimento GPU vs CPU correspondiente.
4. Descarga los resultados (`resultados/entrega3/*.csv`, `*.json`, `*.onnx`) para incorporarlos de vuelta al repositorio local.

| Notebook | Qué compara |
|---|---|
| `M1_cuDF_vs_Polars_Dask.ipynb` | ETL: Pandas vs Polars vs Dask vs cuDF |
| `M2_Clustering_GPU_vs_CPU.ipynb` | K-Means: Scikit-Learn (CPU) vs cuML (GPU) + UMAP |
| `M3_Modelo_Avanzado_GPU.ipynb` | Modelo de recomendación: ALS (baseline) vs NCF (avanzado) |

> **Importante:** al subir un `.zip` de datos generado en Windows, usa el módulo `zipfile` de Python (no `Compress-Archive` de PowerShell) para evitar que los separadores de ruta (`\`) queden como texto literal en vez de como jerarquía de carpetas al extraerse en Linux/Colab. Ver `m1_escalabilidad_entrega3.md` para el detalle de este problema real.

## Dashboard

### Modo interactivo (local)

```bash
cd Codigos/dashboard
python app.py
```
Abre `http://127.0.0.1:8050`.

### Modo contenedor (Docker, reproducible desde cero)

```bash
cd Codigos
docker-compose up --build
```
Abre `http://localhost:8050`.

El dashboard tiene 5 pestañas: exploración de usuarios (M2), interacciones y productos (M1), recomendaciones top-N (M3), rendimiento del pipeline (M4), y prueba de carga (M5).

### Prueba de carga

Con el dashboard corriendo, en otra terminal:

```bash
cd Codigos/dashboard
python -c "from prueba_carga import simular_pico_trafico; import pandas as pd; r = simular_pico_trafico('http://127.0.0.1:8050', list(range(200)), [1,10,50,100,200]); r.to_csv('../resultados/entrega3/prueba_carga.csv', index=False); print(r)"
```

Simula tráfico concurrente creciente (hasta 200 usuarios simultáneos) y guarda los resultados, visibles luego en la pestaña "Prueba de carga" del propio dashboard.

## Herramientas principales

Polars · Dask · PyArrow · Scikit-Learn · Implicit (ALS) · PyTorch (NCF) · RAPIDS cuDF/cuML (GPU) · UMAP · SciPy · Dash/Plotly · Docker · cProfile · threadpoolctl

## Índice de documentación

**Entrega 2:**
- [`Generales/modelado_M3.md`](Generales/modelado_M3.md) — arquitectura y evaluación del modelo baseline ALS
- [`Generales/analisis_rendimiento_M4.md`](Generales/analisis_rendimiento_M4.md) — speedup por etapa, Amdahl
- [`Generales/gestion_de_problemas_M5.md`](Generales/gestion_de_problemas_M5.md) — integración del dashboard

**Entrega 3:**
- [`Generales/m1_escalabilidad_entrega3.md`](Generales/m1_escalabilidad_entrega3.md) — escalabilidad fuerte y débil del ETL
- [`Generales/m1_comparativa_final_entrega3.md`](Generales/m1_comparativa_final_entrega3.md) — Pandas vs Polars vs Dask vs cuDF
- [`Generales/m2_clustering_gpu_entrega3.md`](Generales/m2_clustering_gpu_entrega3.md) — K-Means CPU vs GPU, UMAP
- [`Generales/m3_resultados_finales_entrega3.md`](Generales/m3_resultados_finales_entrega3.md) — NCF vs ALS
- [`Generales/m4_rendimiento_sistema_entrega3.md`](Generales/m4_rendimiento_sistema_entrega3.md) — sistema integrado, latencia, balance de carga
- [`Generales/m5_prueba_carga_entrega3.md`](Generales/m5_prueba_carga_entrega3.md) — simulación de pico de tráfico

## Limitaciones conocidas

- El módulo `analisis_eda.py` (M2, EDA base) no se refactorizó a funciones parametrizables tan exhaustivamente como los demás módulos — ver nota de alcance en el informe de Entrega 2.
- El entrenamiento de NCF (M3) usa un conjunto fijo de negativos por época en vez de resamplearlos (protocolo original de He et al., 2017); la pérdida de entrenamiento cercana a cero sugiere riesgo de sobreajuste — ver `m3_resultados_finales_entrega3.md` para el detalle y la solución propuesta.
- Las mediciones de balance de carga y comunicación (M4) se hicieron sobre una sola operación (deduplicación) en un clúster local de un solo nodo; no se ha medido en un clúster distribuido real multi-nodo.
