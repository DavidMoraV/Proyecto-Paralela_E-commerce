# Dockerfile del dashboard (M5)
# Construye una imagen liviana solo con lo necesario para servir el dashboard.
# Los datos (datos/, resultados/) NO se copian a la imagen -- se montan como
# volúmenes desde docker-compose.yml, para no tener que reconstruir la imagen
# cada vez que cambian los resultados.

FROM python:3.11-slim

WORKDIR /app/dashboard

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8050

# DASH_HOST=0.0.0.0 se define en docker-compose.yml (necesario para que el
# puerto expuesto sea accesible desde fuera del contenedor).
CMD ["python", "app.py"]
