# limpiar_duplicados.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15" (NO dentro de Codigos).
# Mueve (no borra) los duplicados sueltos en la raiz a una carpeta aparte,
# para que los revises antes de eliminarlos definitivamente.

$ErrorActionPreference = "Stop"

$destino = "_duplicados_para_revisar"
New-Item -ItemType Directory -Force -Path $destino | Out-Null

Write-Host "Moviendo carpetas duplicadas..."
$carpetas = @("data", "notebooks", "resultados", "src")
foreach ($c in $carpetas) {
    if (Test-Path $c) {
        Move-Item -Force $c "$destino\$c"
        Write-Host "  Movida: $c"
    } else {
        Write-Host "  (no existe, se omite): $c"
    }
}

Write-Host "Moviendo archivos sueltos duplicados..."
$archivos = @(
    "M2_Analisis_EDA.ipynb",
    "PipelineM1.ipynb",
    "adquisicion_datos_preprocesamiento_M1.md",
    "analisis_exploratorio_explicativo.docx",
    "requirements_m3.txt",
    "user_segments.parquet"
)
foreach ($a in $archivos) {
    if (Test-Path $a) {
        Move-Item -Force $a "$destino\$a"
        Write-Host "  Movido: $a"
    } else {
        Write-Host "  (no existe, se omite): $a"
    }
}

Write-Host ""
Write-Host "Listo. Todo quedo en: $destino"
Write-Host "Revisa esa carpeta; si confirmas que no hay nada que rescatar, borrala manualmente."
Write-Host ""
Write-Host "Estructura final de la raiz:"
Get-ChildItem | Select-Object Name
