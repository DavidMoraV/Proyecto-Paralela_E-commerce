# sincronizar_todo.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15".
# Limpia los duplicados viejos del repo y sube todo el trabajo pendiente de Entrega 3.

$ErrorActionPreference = "Stop"
cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

# Paso 1: confirmar que _duplicados_para_revisar esta en .gitignore
$contenidoGitignore = Get-Content ".gitignore" -Raw
if ($contenidoGitignore -notmatch "_duplicados_para_revisar") {
    Add-Content ".gitignore" "`n# Carpeta de limpieza (duplicados movidos, no se suben)`n_duplicados_para_revisar/"
    Write-Host "Se agrego _duplicados_para_revisar/ al .gitignore"
} else {
    Write-Host ".gitignore ya tenia la exclusion de _duplicados_para_revisar"
}

# Paso 2: stage de TODO (nuevos, modificados, y las eliminaciones de los duplicados viejos)
git add -A

Write-Host ""
Write-Host "=== Esto es lo que se va a subir (revisalo con cuidado) ==="
git status --short

Write-Host ""
Write-Host "Si ves algo de Datasets/ o Codigos/datos/ en la lista de arriba, DETENTE y avisame."
Write-Host "Presiona Enter para continuar con el commit y push, o Ctrl+C para cancelar."
Read-Host

git commit -m "Entrega 3: escalabilidad M1, limpieza de duplicados del repo, estructura entrega3"
git push origin main

Write-Host ""
Write-Host "Sincronizacion completa."
