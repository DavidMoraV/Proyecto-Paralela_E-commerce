# reorganizar_estructura.ps1
# Corre esto DENTRO de la carpeta Codigos.
# Reorganiza los archivos sueltos en la estructura que pide la rubrica de M5:
# dashboard/, src/, notebooks/, datos/, resultados/

$ErrorActionPreference = "Stop"

Write-Host "Creando carpetas..."
New-Item -ItemType Directory -Force -Path "src" | Out-Null
New-Item -ItemType Directory -Force -Path "notebooks" | Out-Null

Write-Host "Moviendo modulos .py a src/ ..."
Move-Item -Force "pipeline_datos.py" "src\pipeline_datos.py"
Move-Item -Force "analisis_eda.py" "src\analisis_eda.py"
Move-Item -Force "modelo_recomendacion.py" "src\modelo_recomendacion.py"
Move-Item -Force "benchmarks.py" "src\benchmarks.py"

Write-Host "Moviendo notebooks a notebooks/ ..."
Move-Item -Force "PipelineM1.ipynb" "notebooks\PipelineM1.ipynb"
Move-Item -Force "M2_Analisis_EDA.ipynb" "notebooks\M2_Analisis_EDA.ipynb"
Move-Item -Force "M3_Modelo_Recomendacion.ipynb" "notebooks\M3_Modelo_Recomendacion.ipynb"
Move-Item -Force "M4_Analisis_Rendimiento.ipynb" "notebooks\M4_Analisis_Rendimiento.ipynb"

Write-Host "Renombrando processed/ a datos/ ..."
Move-Item -Force "processed" "datos"

Write-Host "Limpiando __pycache__ (se regenera solo) ..."
Remove-Item -Recurse -Force "__pycache__" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Listo. Estructura final:"
Get-ChildItem -Directory | Select-Object Name
Write-Host ""
Write-Host "IMPORTANTE: reemplaza ahora los 4 notebooks y app.py / generar_dashboard_html.py"
Write-Host "por las versiones actualizadas que te acabo de compartir (ya tienen las rutas ajustadas)."
