# actualizar_repo.ps1
# Script reutilizable para sincronizar cambios con GitHub.
# Corre esto DENTRO de la carpeta raiz "clase 15" cada vez que quieras subir avances.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File actualizar_repo.ps1
#   powershell -ExecutionPolicy Bypass -File actualizar_repo.ps1 -Mensaje "Limpieza de comentarios en pipeline_datos.py"

param(
    [string]$Mensaje = ""
)

$ErrorActionPreference = "Stop"
cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

git add -A

Write-Host ""
Write-Host "=== Cambios detectados ==="
git status --short

$cambios = git status --short
if (-not $cambios) {
    Write-Host ""
    Write-Host "No hay cambios pendientes. Nada que subir."
    exit 0
}

Write-Host ""
Write-Host "Verifica que NO aparezca Datasets/, Codigos/datos/, ni ningun .zip."

if (-not $Mensaje) {
    $Mensaje = Read-Host "Escribe el mensaje del commit (o Ctrl+C para cancelar)"
}

git commit -m "$Mensaje"
git push origin main

Write-Host ""
Write-Host "Listo. Repositorio actualizado."
