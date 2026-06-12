param(
    [string]$EnvName = "py312"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$conda = "E:\miniconda\Scripts\conda.exe"
if (Test-Path $conda) {
    & $conda run -n $EnvName python run_yj_studio.py
} else {
    python run_yj_studio.py
}
