param(
    [string]$ServerUrl = "",
    [int]$TimeoutSec = 10
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Invoke-JsonUtf8([string]$Uri, [int]$Depth) {
    $request = [System.Net.WebRequest]::Create($Uri)
    $request.Timeout = $TimeoutSec * 1000
    $response = $request.GetResponse()
    try {
        $stream = $response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        try {
            $text = $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } finally {
        $response.Dispose()
    }
    $text | ConvertFrom-Json | ConvertTo-Json -Depth $Depth
}

if (-not $ServerUrl) {
    $Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $ConfigPath = Join-Path $Root "local\config\local.yaml"
    if (-not (Test-Path $ConfigPath)) {
        $ConfigPath = Join-Path $Root "local\config\local.example.yaml"
    }
    $ServerUrl = (Select-String -Path $ConfigPath -Pattern "^\s*server_url\s*:\s*(.+?)\s*$" | Select-Object -First 1).Matches.Groups[1].Value
}

$base = $ServerUrl.TrimEnd("/")

Write-Host "Checking $base/health"
Invoke-JsonUtf8 "$base/health" 6

Write-Host "Checking $base/volumes"
Invoke-JsonUtf8 "$base/volumes" 8
