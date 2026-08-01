# preparar_entrega3.ps1
# Corre esto DENTRO de la carpeta Codigos.
# Crea la estructura nueva para Entrega 3, sin tocar nada de lo que ya
# existe de Entrega 1/2.

$ErrorActionPreference = "Stop"

Write-Host "Creando carpetas nuevas de Entrega 3..."

# Carpetas nuevas
$carpetas = @(
    "colab",
    "modelos",
    "modelos\als_baseline",
    "resultados\entrega3"
)

foreach ($carpeta in $carpetas) {
    New-Item -ItemType Directory -Force -Path $carpeta | Out-Null
    Write-Host "  OK  $carpeta"
}

# .gitkeep en las carpetas que empiezan vacias, para que Git las trackee
# (Git no versiona carpetas vacias, solo archivos)
$gitkeeps = @(
    "colab\.gitkeep",
    "modelos\.gitkeep",
    "modelos\als_baseline\.gitkeep",
    "resultados\entrega3\.gitkeep"
)

foreach ($archivo in $gitkeeps) {
    if (-not (Test-Path $archivo)) {
        New-Item -ItemType File -Force -Path $archivo | Out-Null
    }
}

Write-Host ""
Write-Host "Estructura de Entrega 3 lista. Resumen:"
Write-Host ""
Get-ChildItem -Recurse -Directory | Where-Object {
    $_.FullName -match "colab$|modelos$|als_baseline$|entrega3$"
} | ForEach-Object { Write-Host "  $($_.FullName)" }

Write-Host ""
Write-Host "Nada de Entrega 1/2 fue modificado (src/, notebooks/, dashboard/, datos/, resultados/*.csv siguen igual)."
