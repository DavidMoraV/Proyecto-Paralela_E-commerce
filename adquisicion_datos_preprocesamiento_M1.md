## Adquisición de Datos y Preprocesamiento (M1)

### Descripción del dataset

El proyecto utiliza el RetailRocket E-commerce Dataset, publicado en Kaggle, compuesto por tres fuentes:

| Archivo | Contenido | Filas (verificadas) |
|---|---|---|
| `events.csv` | Eventos de interacción usuario-producto: vista, adición al carrito, transacción | 2,756,101 |
| `category_tree.csv` | Árbol de categorías de producto (relación hijo-padre) | 1,669 |
| `item_properties_part1.csv` + `part2.csv` | Propiedades de producto (categoría, disponibilidad, precio, marca, etc.) a lo largo del tiempo | ~20 millones (combinado) |

El esquema de `events.csv`, la tabla central del pipeline, es:

| Columna | Tipo | Descripción |
|---|---|---|
| `timestamp` | entero (ms desde epoch) | Momento del evento |
| `visitorid` | entero | Identificador de usuario |
| `event` | texto | `view`, `addtocart` o `transaction` |
| `itemid` | entero | Identificador de producto |
| `transactionid` | entero (nulo si no aplica) | Identificador de transacción, solo presente si `event = transaction` |

### Diseño del pipeline

El pipeline (implementado en Polars, con Dask para el particionado distribuido) sigue cuatro etapas:

1. **Ingesta** (`ingestar_*`): lectura perezosa (*lazy*) de cada fuente, con detección automática de separador (`,`/`;`) y formato decimal, para tolerar archivos que hayan sido reformateados por herramientas externas (ver "Gestión de problemas").
2. **Limpieza** (`limpiar_*`): eliminación de duplicados exactos, filtrado de valores nulos en `itemid`/`visitorid`, validación de que `event` tome solo los tres valores esperados. Incluye una verificación automática de calidad que detecta cuándo la deduplicación podría estar eliminando datos legítimos por baja cardinalidad del timestamp.
3. **Transformación** (`transformar_eventos`): construcción de variables derivadas para las etapas siguientes — `peso_implicito` (confianza de la señal de interacción: view=1, addtocart=3, transaction=5, insumo directo del modelo ALS de M3), y descomposición temporal (`fecha`, `hora_del_dia`, `dia_semana`) para el EDA de M2.
4. **Exportación** (`exportar_parquet`): salida en formato Parquet particionado por día de la semana, reduciendo el I/O de lectura para los módulos posteriores.

### Validación inicial — Benchmark Pandas vs Polars vs Dask

Se comparó el tiempo de carga y limpieza de `events.csv` completo (2,755,641 filas tras limpieza) entre las tres herramientas evaluadas en la justificación técnica:

| Herramienta | Tiempo (s) | Memoria pico (MB) | Speedup vs. Pandas |
|---|---|---|---|
| Pandas (baseline secuencial) | 4.90 | 129.5 | 1.00x |
| **Polars** | **1.62** | 454.3 | **3.02x** |
| Dask (4 particiones) | 4.01 | 102.3 | 1.22x |

<img width="1715" height="558" alt="image" src="https://github.com/user-attachments/assets/6be26e53-eb17-40e4-8fd8-21cc0c7b9749" />


**Interpretación:** Polars fue consistentemente la opción más rápida, a costa de un mayor uso de memoria — un trade-off razonable dado que el proyecto no opera bajo restricción severa de RAM. El resultado más interesante es el de Dask: en una prueba preliminar con una muestra de ~900,000 filas, Dask resultó *más lento* que Pandas (0.42x), pero al escalar al dataset completo de 2.76M filas, Dask superó a Pandas (1.22x). Esto confirma empíricamente el principio señalado en el Marco Teórico: el costo de coordinación de tareas distribuidas de Dask solo se amortiza a partir de cierto volumen de datos, y no es la herramienta más eficiente para datasets que ya caben cómodamente en memoria de un solo proceso.

### Gestión de problemas

Se documentan tres obstáculos de calidad de datos encontrados y resueltos durante esta etapa, con causa raíz identificada y verificación cuantitativa de la solución — ver sección "Gestión de problemas" adjunta.

### Reproducibilidad

El pipeline completo es ejecutable de principio a fin desde `src/pipeline_datos.py` o desde el notebook `M1_pipeline_datos.ipynb`, con instrucciones detalladas en el `README.md` del repositorio.
