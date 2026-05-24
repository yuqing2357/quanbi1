#requires -Version 5.0
<#
.SYNOPSIS
    Copy SAM3 source + weights from D:\商书记项目\sam3\sam3\ into the project.

.DESCRIPTION
    YJ Studio's AI Dock looks for SAM3 in two places inside this repo:
      - libs\sam3\          (Python package — model_builder.py, model/, agent/, ...)
      - weights\sam3.pt     (model checkpoint, ~5 GB)

    The original assets live outside the project at:
      D:\商书记项目\sam3\sam3\sam3\     ← the inner-most "sam3" is the Python pkg
      D:\商书记项目\sam3\sam3\weights\  ← all checkpoints

    This script copies them in with robocopy, skipping .git and large eval
    toolkits we don't need at inference time.

.PARAMETER SkipWeights
    Don't copy the 5 GB sam3.pt. Useful when you only want to refresh the
    source code.

.PARAMETER SourceRoot
    Override the upstream SAM3 git checkout location.

.EXAMPLE
    PS> .\tools\copy_sam3_assets.ps1

.EXAMPLE
    PS> .\tools\copy_sam3_assets.ps1 -SkipWeights
#>

param(
    [string] $SourceRoot = 'D:\商书记项目\sam3\sam3',
    [switch] $SkipWeights
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$SourcePkg = Join-Path $SourceRoot 'sam3'        # inner sam3 — the python package
$SourceWeights = Join-Path $SourceRoot 'weights'

$DestPkg = Join-Path $ProjectRoot 'libs\sam3'
$DestWeights = Join-Path $ProjectRoot 'weights'

Write-Host "Source pkg     : $SourcePkg"
Write-Host "Source weights : $SourceWeights"
Write-Host "Dest pkg       : $DestPkg"
Write-Host "Dest weights   : $DestWeights"
Write-Host ""

if (-not (Test-Path -LiteralPath $SourcePkg)) {
    throw "SAM3 source package not found at $SourcePkg"
}
if (-not $SkipWeights -and -not (Test-Path -LiteralPath $SourceWeights)) {
    throw "SAM3 weights folder not found at $SourceWeights"
}

# robocopy returns 0-7 for success; 8+ are failures. Run it via cmd /c to
# preserve those exit codes in PowerShell's own $LASTEXITCODE.
Write-Host "[1/2] Copying sam3 Python package (skipping .git, __pycache__, eval toolkits)..."
$pkgArgs = @(
    "`"$SourcePkg`"", "`"$DestPkg`"",
    '/E',
    '/XD', '.git', '__pycache__', 'hota_eval_toolkit', 'teta_eval_toolkit',
    '/NFL', '/NDL', '/NP'
)
cmd /c "robocopy $($pkgArgs -join ' ')" | Out-Host
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed for sam3 source (exit $LASTEXITCODE)"
}

if (-not $SkipWeights) {
    Write-Host ""
    Write-Host "[2/2] Copying weights (this is the slow part, ~5 GB)..."
    $weightArgs = @(
        "`"$SourceWeights`"", "`"$DestWeights`"",
        '/E', '/NFL', '/NDL'
    )
    cmd /c "robocopy $($weightArgs -join ' ')" | Out-Host
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for weights (exit $LASTEXITCODE)"
    }
} else {
    Write-Host ""
    Write-Host "[2/2] Skipped weights (-SkipWeights)"
}

Write-Host ""
Write-Host "Done. Quick verification:"
if (Test-Path -LiteralPath (Join-Path $DestPkg '__init__.py')) {
    Write-Host "  [OK] libs\sam3\__init__.py exists"
} else {
    Write-Host "  [!!] libs\sam3\__init__.py NOT found — check the copy"
}
if (-not $SkipWeights) {
    $ckpt = Join-Path $DestWeights 'sam3.pt'
    if (Test-Path -LiteralPath $ckpt) {
        $size = (Get-Item -LiteralPath $ckpt).Length / 1GB
        Write-Host ("  [OK] weights\sam3.pt present ({0:N2} GB)" -f $size)
    } else {
        Write-Host "  [!!] weights\sam3.pt NOT found — check the copy"
    }
}
