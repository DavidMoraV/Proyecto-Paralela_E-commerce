# sincronizar_entrega3.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15".
# Sube todo el trabajo nuevo de Entrega 3 (M1, M3, M4, M5, y ahora M2-GPU),
# excluyendo datos pesados y archivos temporales.

$ErrorActionPreference = "Stop"
cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

# Paso 1: asegurar que los .zip no se suban (son artefactos temporales para Colab)
$contenidoGitignore = Get-Content ".gitignore" -Raw
if ($contenidoGitignore -notmatch "\*\.zip") {
    Add-Content ".gitignore" "`n# Archivos .zip temporales (para subir a Colab, no se versionan)`n*.zip"
    Write-Host "Se agrego *.zip al .gitignore"
} else {
    Write-Host ".gitignore ya excluia los .zip"
}

# Paso 2: stage de todo
git add -A

Write-Host ""
Write-Host "=== Esto es lo que se va a subir (revisalo con cuidado) ==="
git status --short

Write-Host ""
Write-Host "Verifica en la lista de arriba que NO aparezca:"
Write-Host "  - Datasets/ ni Codigos/datos/ (datos pesados)"
Write-Host "  - eventos_limpios.zip (en ninguna ubicacion)"
Write-Host "  - __pycache__/"
Write-Host ""
Write-Host "Presiona Enter para continuar con el commit y push, o Ctrl+C para cancelar."
Read-Host

git commit -m "Entrega 3: M1 (escalabilidad + cuDF), M2 (clustering GPU), M3 (NCF), M4 (sistema integrado), M5 (docker + prueba de carga)"
git push origin main

Write-Host ""
Write-Host "Sincronizacion completa."
