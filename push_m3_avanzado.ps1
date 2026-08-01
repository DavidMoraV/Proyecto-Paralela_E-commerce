# push_m3_avanzado.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15" (la que tiene .git).

$ErrorActionPreference = "Stop"

cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

# Verificar que los archivos nuevos existen antes de agregarlos
$archivos = @(
    "Codigos\src\modelo_avanzado.py",
    "Codigos\colab\M3_Modelo_Avanzado_GPU.ipynb"
)
foreach ($a in $archivos) {
    if (-not (Test-Path $a)) {
        Write-Host "FALTA: $a -- colocalo antes de continuar." -ForegroundColor Red
        exit 1
    }
}

git add Codigos\src\modelo_avanzado.py
git add Codigos\colab\M3_Modelo_Avanzado_GPU.ipynb

Write-Host ""
Write-Host "Archivos preparados para commit:"
git status --short

Write-Host ""
git commit -m "Entrega 3: modelo avanzado NCF (modulo + notebook Colab)"
git push origin main

Write-Host ""
Write-Host "Push completado."
