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

# M3 — Modelo de Machine Learning y Recomendación

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