# Makefile — Sistema de Recomendación Paralelo (RetailRocket)
# Ejecuta cada etapa del pipeline de forma reproducible.
#
# NOTA PARA WINDOWS: 'make' no viene instalado por defecto. Opciones:
#   - Git Bash (incluye 'make' si instalaste "Git for Windows" con opciones completas)
#   - choco install make   (si tienes Chocolatey)
#   - O simplemente correr los comandos de cada target manualmente en PowerShell
#     (cada uno es una sola línea, están comentados abajo para copiar/pegar)

PYTHON := python

.PHONY: install m1 m2 m3 m4 dashboard dashboard-html all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

# Corre el pipeline ETL completo (M1) desde la línea de comandos
m1:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/PipelineM1.ipynb

# Corre el EDA y clustering (M2)
m2:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/M2_Analisis_EDA.ipynb

# Corre el entrenamiento y evaluación del modelo (M3)
m3:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/M3_Modelo_Recomendacion.ipynb

# Corre el análisis de rendimiento (M4)
m4:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/M4_Analisis_Rendimiento.ipynb

# Levanta el dashboard interactivo (Dash, requiere servidor activo)
dashboard:
	cd dashboard && $(PYTHON) app.py

# Genera el dashboard HTML estático (no requiere servidor)
dashboard-html:
	cd dashboard && $(PYTHON) generar_dashboard_html.py

# Corre todo el pipeline de punta a punta, en orden, y genera el dashboard HTML
all: m1 m2 m3 m4 dashboard-html
	@echo "Pipeline completo. Abre dashboard/dashboard.html"

clean:
	rm -rf src/__pycache__ notebooks/.ipynb_checkpoints
