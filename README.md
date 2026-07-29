# M1 Pipeline de Datos (Entregable 2)

Módulo de Ingeniería de Datos del Sistema de Recomendación Paralelo para E-Commerce.
Responsable: David Mora.

## Qué hace

Ingesta, limpia, transforma y exporta a Parquet las tres fuentes del dataset
RetailRocket (`events.csv`, `category_tree.csv`, `item_properties_part*`),
usando Polars como motor principal (paralelismo multinúcleo) y Dask para
particionado distribuido. Detecta y corrige automáticamente archivos que
hayan sido reformateados por Excel (separador `;`, decimales con `,`,
timestamps en notación científica) — ver "Gestión de problemas" en el
informe para el detalle de este caso real encontrado durante el desarrollo.

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura

```
proyecto/
├── src/
│   └── pipeline_datos.py       # Módulo principal: ingesta, limpieza, transformación, exportación
├── M1_pipeline_datos.ipynb     # Notebook equivalente, con validación y benchmark paso a paso
├── benchmark_etl.py            # Benchmark Pandas vs Polars vs Dask (standalone)
├── data/
│   ├── raw/                    # Coloca aquí los 4 archivos originales de RetailRocket
│   └── processed/              # Salida en Parquet (generada al correr el pipeline)
├── resultados/
│   └── benchmark_etl.csv       # Resultado del benchmark más reciente
└── requirements.txt
```

## Cómo correrlo

1. Coloca en `data/raw/` los archivos originales de RetailRocket:
   `events.csv`, `category_tree.csv`, `item_properties_part1.csv` (o `.xlsx`),
   `item_properties_part2.csv` (o `.xlsx`).

2. Pipeline completo (genera los Parquet para M2/M3):

   ```bash
   python -m src.pipeline_datos
   ```

   O, de forma interactiva con explicación paso a paso, abre y corre
   `M1_pipeline_datos.ipynb` en Jupyter.

3. Benchmark de validación (Pandas vs Polars vs Dask):

   ```bash
   python benchmark_etl.py --carpeta data/raw
   ```

## Notas de calidad de datos

Si al correr el pipeline aparece un mensaje `ALERTA DE CALIDAD` en el log,
significa que el archivo de entrada tiene una cardinalidad de `timestamp`
anormalmente baja (síntoma de haber pasado por Excel), lo que puede causar
que se eliminen eventos legítimos como falsos duplicados. En ese caso,
volver a descargar el archivo original desde Kaggle sin abrirlo en Excel en
ningún punto intermedio, y verificar con un editor de texto plano (no Excel)
que el separador sea `,` y el timestamp un entero completo antes de
reprocesar.

#  Modelo de Machine Learning y Recomendación

Módulo de ML del Sistema de Recomendación Paralelo para E-Commerce.
Responsable: Siloé Campos

## Qué hace

Implementa y compara dos enfoques de recomendación sobre el dataset RetailRocket:
- **BaselineModel**: filtrado colaborativo mediante ALS (Alternating Least Squares), vía la librería `implicit`.
- **NeuralModel**: red neuronal NCF (Neural Collaborative Filtering), vía PyTorch.

Ambos se evalúan con el mismo protocolo (NDCG@10, MAP@10, Hit Rate@10, sobre 1 positivo + 99 negativos muestreados por usuario), permitiendo una comparación justa.

## Requisitos

Este módulo requiere un entorno **separado** del resto del pipeline (M1/M2), debido a incompatibilidades de PyTorch con Mac Intel + Python 3.14 (ver sección "Problemas conocidos"):

```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements_m3.txt
```

## Cómo correrlo

Requiere que `data/processed/eventos_limpios.parquet` ya exista (generado por el pipeline de M1).

1. Verificar los datos de entrada:
```bash
   python src/verificar_datos.py
```

2. Preparar datos para el modelo (split train/test + matrices):
```bash
   python src/preparar_datos_modelo.py
```

3. Preparar pares de entrenamiento para el modelo neural (con muestreo negativo):
```bash
   python src/preparar_datos_neural.py
```

4. Entrenar ambos modelos (IMPORTANTE: fijar `OPENBLAS_NUM_THREADS=1` para evitar degradación de rendimiento 3-4x en ALS — ver "Problemas conocidos"):
```bash
   export OPENBLAS_NUM_THREADS=1
   caffeinate -i python src/modelo_recomendacion.py   # caffeinate solo en Mac, evita que la laptop entre en reposo
```
   Tiempo estimado: ALS ~95s, NCF (5 épocas) ~66 min en CPU.

5. Evaluar y comparar ambos modelos:
```bash
   python src/evaluar_modelo_neural.py
```

6. Generar tabla y gráficas comparativas:
```bash
   python src/generar_reporte_comparativo.py
```

## Resultados (Entrega 2)

| Métrica | ALS (Baseline) | NCF (Neural) |
|---|---|---|
| NDCG@10 | 0.499 | 0.394 |
| MAP@10 | 0.450 | 0.318 |
| Hit Rate@10 | 0.658 | 0.641 |
| Tiempo entrenamiento | 95s | 3945s (5 épocas) |

## Problemas conocidos

Ver sección "Gestión de problemas" del informe IEEE para el detalle completo. Resumen:
1. PyTorch no publica wheels para Mac Intel + Python 3.14 → requiere Python 3.12 y `torch==2.2.2` (última versión con soporte x86_64 en Mac).
2. Requiere `numpy==1.26.4` y `scipy==1.12.0` fijos (compatibilidad de ABI con torch 2.2.2).
3. `OPENBLAS_NUM_THREADS=1` es obligatorio antes de entrenar con `implicit` — sin esto, el entrenamiento de ALS se degrada ~3.4x.
4. El backend `mps` de PyTorch se cuelga silenciosamente en Mac Intel con capas `nn.Embedding` — el código fuerza CPU explícitamente.
5. El dataset tiene cold-start severo (85.8% de usuarios con 1 sola interacción en train), lo cual limita el techo de ambos modelos y favorece a ALS sobre NCF en esta etapa de entrenamiento.
=======
# Sistema de Recomendación Paralelo para E-Commerce

Proyecto del curso **Computación Paralela y Distribuida** 

Sistema de recomendación de productos construido sobre el dataset [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (2,755,641 eventos, 1,407,580 usuarios), con énfasis en demostrar y medir el uso de computación paralela y distribuida en cada etapa del pipeline: ingesta, EDA, modelado y análisis de rendimiento.

**Entrega actual: Entrega 2 — Implementación Inicial y Validación Parcial.**
El informe completo está en [`Generales/entregable 2_Sistema_de_Recomendación_Paralelo_para_E_Commerce.pdf`](<Generales/entregable%202_Sistema_de_Recomendación_Paralelo_para_E_Commerce.pdf>).

## Equipo

Ana María Ramírez (M2) · María José (M2) · Siloé Campos · Julio · David Mora (M1, M3, M4, M5)

## Arquitectura

```
Datos crudos (RetailRocket)
        │
        ▼
  M1 · ETL (Polars + Dask)              → datos/*.parquet
        │
        ▼
  M2 · EDA y segmentación (K-Means)     → datos/user_segments.parquet
        │
        ▼
  M3 · Modelo de recomendación (ALS)    → resultados/recomendaciones_muestra.csv
        │
        ├──► M4 · Análisis de rendimiento (speedup, Amdahl, profiling)
        │
        ▼
  M5 · Dashboard (Dash / HTML estático)
```

## Resultados clave (dataset completo)

| Métrica | Valor |
|---|---|
| Eventos procesados | 2,755,641 |
| Usuarios únicos | 1,407,580 |
| Speedup ETL (Polars vs Pandas) | 3.02× |
| Hit Rate@10 del modelo | 0.0266 |
| Speedup máx. entrenamiento (ALS) | 1.09× (fracción secuencial 89.2%, Ley de Amdahl) |

Detalle completo de metodología, benchmarks y hallazgos en el informe PDF.

## Estructura del repositorio

```
.
├── Datasets/                  # Datos crudos de RetailRocket (NO incluido — ver "Cómo obtener los datos")
├── Generales/                 # Documentos del curso e informes del equipo
│   └── entregable 2_...pdf    # Informe IEEE completo de la Entrega 2
└── Codigos/
    ├── src/                   # Módulos Python reutilizables
    │   ├── pipeline_datos.py       # M1 — ingesta, limpieza, transformación, exportación
    │   ├── analisis_eda.py         # M2 — features de usuario, clustering
    │   ├── modelo_recomendacion.py # M3 — matriz de interacciones, ALS, evaluación
    │   └── benchmarks.py           # M4 — speedup, Amdahl, profiling
    ├── notebooks/             # Un notebook por módulo, ejecutables de punta a punta
    │   ├── PipelineM1.ipynb
    │   ├── M2_Analisis_EDA.ipynb
    │   ├── M3_Modelo_Recomendacion.ipynb
    │   └── M4_Analisis_Rendimiento.ipynb
    ├── dashboard/
    │   ├── app.py                     # Dashboard interactivo (Dash)
    │   └── generar_dashboard_html.py  # Dashboard estático (HTML autocontenido)
    ├── datos/                 # Parquet procesados (NO incluido — se regenera con M1/M2)
    ├── resultados/             # CSV/PNG de benchmarks y métricas (SÍ incluido, es evidencia)
    ├── Makefile
    └── requirements.txt
```

## Cómo obtener los datos

El dataset no se incluye en el repositorio (ver `.gitignore`). Para reproducir el proyecto:

1. Descarga el dataset desde [Kaggle — RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
2. Coloca los 4 archivos (`events.csv`, `category_tree.csv`, `item_properties_part1.csv`/`.xlsx`, `item_properties_part2.csv`/`.xlsx`) dentro de `Datasets/`.

> **Nota:** el pipeline detecta y corrige automáticamente archivos que hayan pasado por Excel (separador `;`, decimales con `,`, timestamps en notación científica) — ver "Gestión de problemas" en el informe para el detalle de este caso real encontrado durante el desarrollo.

## Instalación y ejecución

```bash
pip install -r Codigos/requirements.txt
```

Ejecutar en orden (cada notebook alimenta al siguiente):

```bash
cd Codigos
jupyter nbconvert --to notebook --execute --inplace notebooks/PipelineM1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M2_Analisis_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M3_Modelo_Recomendacion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/M4_Analisis_Rendimiento.ipynb
```

O con el `Makefile` (Linux/Mac/Git Bash; en Windows sin `make`, correr los comandos de arriba manualmente):

```bash
make all
```

### Dashboard

```bash
cd Codigos/dashboard
python app.py                      # interactivo, http://127.0.0.1:8050
# o
python generar_dashboard_html.py   # genera dashboard.html, sin servidor
```

## Herramientas principales

Polars · Dask · PyArrow · Scikit-Learn · Implicit (ALS) · SciPy · Dash / Plotly · cProfile · threadpoolctl

## Documentación adicional

- [`Generales/entrega_M3_para_M4.md`](Generales/entrega_M3_para_M4.md) — insumos de M3 para el análisis de rendimiento de M4
- [`Generales/modelado_M3.md`](Generales/modelado_M3.md) — detalle del modelo de recomendación
- [`Generales/analisis_rendimiento_M4.md`](Generales/analisis_rendimiento_M4.md) — metodología y hallazgos de rendimiento
- [`Generales/gestion_de_problemas_M5.md`](Generales/gestion_de_problemas_M5.md) — problemas de integración del dashboard

