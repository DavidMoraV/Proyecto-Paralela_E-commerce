# push_entrega3.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15".
# Sube todo el trabajo pendiente de Entrega 3 (M1, M3, M4, M5) al repositorio.

$ErrorActionPreference = "Stop"
cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

git add -A

Write-Host ""
Write-Host "=== Esto es lo que se va a subir (revisalo con cuidado) ==="
git status --short

Write-Host ""
Write-Host "Verifica en la lista de arriba:"
Write-Host "  - Que NO aparezca nada de Datasets/ ni Codigos/datos/"
Write-Host "  - Que SI aparezcan: analisis_sistema.py, M4_Rendimiento_Sistema.ipynb,"
Write-Host "    M1_cuDF_vs_Polars_Dask.ipynb, docker-compose.yml, Dockerfile,"
Write-Host "    los .md de Generales, y los CSV/JSON/ONNX de resultados/entrega3 y modelos/"
Write-Host ""
Write-Host "Presiona Enter para continuar con el commit y push, o Ctrl+C para cancelar."
Read-Host

git commit -m "Entrega 3: M1 (escalabilidad + cuDF), M3 (NCF vs ALS), M4 (sistema integrado), M5 (docker-compose)"
git push origin main

Write-Host ""
Write-Host "Push completado."
