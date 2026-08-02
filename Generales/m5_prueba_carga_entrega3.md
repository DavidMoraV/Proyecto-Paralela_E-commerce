## M5 — Prueba de Carga: Simulación de Pico de Tráfico (Entrega 3)

### Metodología

Se simuló tráfico concurrente contra el dashboard (Dash) usando un cliente HTTP con `ThreadPoolExecutor`, replicando el formato exacto de las solicitudes que el cliente JavaScript de Dash envía al endpoint `_dash-update-component`. Cada solicitud simula un usuario distinto pidiendo recomendaciones para un `visitorid` aleatorio (sin caché posible entre solicitudes, ya que cada usuario pide algo distinto — el escenario realista de un pico de tráfico). Se midió en niveles crecientes de concurrencia (1, 10, 50, 100, 200 usuarios simultáneos), simulando la rampa de un evento tipo Black Friday.

### Resultados

| Usuarios concurrentes | Latencia media | Latencia p95 | Throughput | Tasa de error |
|---|---|---|---|---|
| 1 | 41.5 ms | 41.5 ms | 23.8 req/s | 0% |
| 10 | 226.8 ms | 299.3 ms | 31.0 req/s | 0% |
| 50 | 957.2 ms | 1,626.2 ms | 28.7 req/s | 0% |
| 100 | 1,903.7 ms | 3,513.5 ms | 27.0 req/s | 0% |
| 200 | 3,566.3 ms | 6,555.2 ms | 28.1 req/s | 0% |

### Hallazgo principal: el throughput se estanca, la latencia crece linealmente

El dato más relevante de esta prueba no es un fallo del sistema (0% de errores en todos los niveles — el servidor nunca se cae ni rechaza solicitudes), sino un **techo de throughput**: independientemente de si hay 1 o 200 usuarios pidiendo recomendaciones al mismo tiempo, el sistema procesa consistentemente entre 27 y 31 solicitudes por segundo. Como consecuencia directa (ley de colas: latencia ≈ concurrencia / throughput), la latencia crece de forma casi lineal con la carga: de 41.5 ms con 1 usuario a **3.57 segundos** con 200 usuarios simultáneos — un tiempo de espera inaceptable para una experiencia de usuario en tiempo real durante un pico de tráfico real.

### Causa raíz: el GIL de Python, no el modelo de concurrencia del servidor

La primera hipótesis fue que el servidor de desarrollo de Flask (usado internamente por Dash) opera con un solo hilo por defecto. Se activó `threaded=True` en `app.run()` para permitir el manejo concurrente de conexiones — pero el throughput **no mejoró** (se mantuvo en el mismo rango de ~25-30 req/s antes y después del cambio).

La causa real es más fundamental: los callbacks del dashboard realizan trabajo intensivo de CPU en Python puro — filtrado de DataFrames de pandas y construcción de figuras de Plotly (serialización a JSON de miles de puntos de datos). Este tipo de trabajo **no libera el GIL** (Global Interpreter Lock) de Python, por lo que, aunque el servidor acepte múltiples conexiones simultáneas (gracias a `threaded=True`), solo un hilo puede ejecutar bytecode de Python a la vez — el cómputo real de cada solicitud sigue siendo serializado, sin importar cuántos hilos estén "escuchando" conexiones. `threaded=True` ayuda a *aceptar* más conexiones concurrentemente, pero no *paraleliza* el trabajo de construir cada respuesta.

Esta es una limitación bien documentada de Python para cargas de trabajo con cómputo intensivo en CPU (a diferencia de cargas de I/O, donde `threaded=True` sí ayuda) — el mismo principio de fondo, aunque en un contexto distinto, que ya se documentó con el GIL y BLAS durante el entrenamiento de ALS en M3 (Entrega 2).

### Recomendación para producción

La solución real no es ajustar el modelo de hilos, sino usar **paralelismo por procesos**, que sí evade el GIL (cada proceso tiene su propio intérprete de Python):

1. **Servidor WSGI de producción con múltiples workers** (ej. `gunicorn --workers 4 app:server`), en vez del servidor de desarrollo de Flask que Dash usa por defecto — el propio Flask advierte explícitamente "no usar en producción" en sus logs.
2. **Múltiples réplicas del contenedor** en `docker-compose.yml`, con un balanceador de carga (ej. Nginx) distribuyendo solicitudes entre ellas — natural dado que ya se cuenta con la imagen Docker de M5.

Ambas opciones multiplican la capacidad de cómputo real disponible (varios procesos, cada uno con su propio GIL), a diferencia de `threaded=True`, que solo ayuda a la capa de aceptación de conexiones.

### Gestión de problemas

**Problema — `threaded=True` no resolvió el cuello de botella esperado.** Se documenta como hallazgo válido, no como una corrección fallida: permitió descartar la hipótesis inicial (servidor de un solo hilo) y aislar la causa real (cómputo CPU-bound serializado por el GIL), lo cual llevó a una recomendación de producción más precisa (paralelismo por procesos, no por hilos) que no se habría identificado sin este paso intermedio.
