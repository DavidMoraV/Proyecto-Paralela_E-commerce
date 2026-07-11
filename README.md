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
