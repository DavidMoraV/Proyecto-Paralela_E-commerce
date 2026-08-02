# commit_final.ps1
# Corre esto DENTRO de la carpeta raiz "clase 15".
# Commit final con todo el contenido pendiente del directorio.

$ErrorActionPreference = "Stop"
cd "E:\Ing ciencia de datos\Septimo cuatri\Computacion paralela\clase 15"

# Confirmar que las exclusiones de siempre siguen en el .gitignore
$contenidoGitignore = Get-Content ".gitignore" -Raw
$reglas = @("Datasets/", "Codigos/datos/", "*.zip", "__pycache__/")
foreach ($regla in $reglas) {
    if ($contenidoGitignore -notmatch [regex]::Escape($regla)) {
        Write-Host "ADVERTENCIA: no se encontro la regla '$regla' en .gitignore -- revisa antes de continuar." -ForegroundColor Yellow
    }
}

# Stage de todo
git add -A

Write-Host ""
Write-Host "=== Esto es lo que se va a subir (revisalo con cuidado) ==="
git status --short

Write-Host ""
Write-Host "Verifica que NO aparezca Datasets/, Codigos/datos/, ni ningun .zip."
Write-Host "Presiona Enter para continuar con el commit y push, o Ctrl+C para cancelar."
Read-Host

git commit -m "Entrega 3: consolidacion final - M2 clustering GPU vs CPU + UMAP, y contenido pendiente del directorio"
git push origin main

Write-Host ""
Write-Host "Commit final completado."
