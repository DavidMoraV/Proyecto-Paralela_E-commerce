## Gestión de problemas (M5 — Entrega 2)

### Problema 1 — Formato de los scores del modelo (M3) ilegible en el dashboard

**Detección:** al conectar el dashboard con `recomendaciones_muestra.csv` (salida de M3), la tabla de recomendaciones mostraba `0,0000` en las 10 filas, y el eje del gráfico usaba una notación con sufijo `n` poco clara para el usuario final.

**Causa raíz:** los scores reales del modelo ALS sobre el dataset completo son del orden de 10⁻⁹–10⁻¹⁰ (productos punto entre vectores latentes sobre una matriz extremadamente dispersa, 0.0006% de densidad). El formato de despliegue original (`.4f`, 4 decimales fijos) redondeaba cualquier valor de esa magnitud a cero, y Plotly aplicaba automáticamente notación de prefijo SI (`n` = nano) al eje, que no es intuitiva fuera de un contexto de ingeniería.

**Solución aplicada:** se cambió el formato de la tabla a notación científica explícita (`.4e`) y se fijó el formato del eje (`yaxis_tickformat=".2e"`) en vez de dejar el autoformato de Plotly. Se agregó además una nota aclaratoria visible en el dashboard: el score de ALS es un producto punto, no una probabilidad, y lo relevante es el orden relativo entre recomendaciones, no la magnitud absoluta.

### Problema 2 — El cluster más activo (M2) opacaba visualmente a los demás en el radar chart

**Detección:** el radar de perfil de clusters mostraba una sola forma dominante (Cluster 2) y el resto de los clusters aparecían aplastados contra el centro, ilegibles.

**Causa raíz:** el Cluster 2 no es superior en una sola variable, sino en las 5 variables de perfil a la vez, con valores 10-12 veces más grandes que el segundo lugar (ej. 121.79 interacciones promedio vs 10.20 del siguiente cluster). Una normalización min-max estándar sobre las 6 series a la vez no resuelve el problema: el cluster outlier sigue llegando a 1.0 en los 5 ejes y el resto queda igual de comprimido, solo que en otra escala.

**Solución aplicada:** se separó la visualización en dos gráficos — uno con los 5 clusters restantes normalizados entre sí (donde sí se distinguen sus diferencias relativas), y otro aparte solo para el Cluster 2 en su propia escala — más una tabla de valores absolutos para no perder la comparación cuantitativa real entre todos los segmentos.

### Problema 3 — Incompatibilidades de rutas al reorganizar la estructura de carpetas

**Detección:** al migrar el proyecto de una estructura de archivos sueltos a la organización que pide la rúbrica (`dashboard/`, `src/`, `notebooks/`, `datos/`, `resultados/`), cada notebook y script falló con una combinación de `ModuleNotFoundError` y `FileNotFoundError` al volver a ejecutarse.

**Causa raíz — dos problemas distintos mezclados:**
1. Los módulos (`pipeline_datos.py`, `analisis_eda.py`, `modelo_recomendacion.py`, `benchmarks.py`) se movieron a `src/`, pero `sys.path.append(str(RAIZ_PROYECTO))` seguía apuntando a la raíz del proyecto, no a la nueva subcarpeta — los notebooks de M3 y M4 no encontraban los módulos de M1 y M2 que importan directamente.
2. La carpeta de datos procesados se renombró de `processed/` a `datos/` (para alinear el nombre con la convención de la rúbrica), pero las rutas `CARPETA_PROCESSED = RAIZ_PROYECTO / "processed"` seguían escritas con el nombre anterior en cada notebook y en `dashboard/app.py` / `dashboard/generar_dashboard_html.py`.
3. Adicionalmente, el notebook de M1 usaba una variable de ruta independiente (`CARPETA`) heredada de una reorganización anterior del proyecto (cuando los datos crudos y procesados vivían en la misma carpeta), que no se había actualizado cuando los datos crudos se separaron en una carpeta `Datasets/` distinta de `Codigos/`.

**Solución aplicada:** se estandarizó la convención de rutas en los cinco módulos: una variable `RAIZ_PROYECTO` fija que apunta a `Codigos/`, con `sys.path.append(str(RAIZ_PROYECTO / "src"))` para las importaciones y `RAIZ_PROYECTO / "datos"` / `RAIZ_PROYECTO / "resultados"` para lectura/escritura de datos. Para M1 específicamente, se separaron explícitamente dos rutas (`CARPETA_DATASETS` para los CSV originales, `CARPETA_CODIGOS` para las salidas procesadas), reflejando la separación real que ya existía en disco entre `Datasets/` y `Codigos/`.

**Lección para el resto del proyecto:** cuando varios módulos desarrollados por distintos integrantes del equipo (M1-M5) dependen de rutas compartidas, un cambio estructural en un módulo (como reorganizar carpetas) se propaga silenciosamente a todos los demás si las rutas no siguen una convención única y centralizada. Se recomienda, para la Entrega 3, definir esta convención una sola vez (por ejemplo, en un archivo de configuración compartido) en vez de repetirla manualmente en cada notebook.
