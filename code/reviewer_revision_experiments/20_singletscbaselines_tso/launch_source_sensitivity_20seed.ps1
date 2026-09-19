param(
    [int]$WaitForPid = 0,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $PackageRoot
$python = (Get-Command python -ErrorAction Stop).Source
$runner = Join-Path $PSScriptRoot 'run_frozen_v22_sources.py'

if ($WaitForPid -gt 0) {
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 20
    }
}

foreach ($source in 1..2) {
    $output = Join-Path $OutputRoot ("source_i" + $source)
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $port = 8830 + $source
    & $python -u $runner `
        --source-intersection $source `
        --difficulty normal `
        --seeds (1..20) `
        --port $port `
        --output-dir $output
    if ($LASTEXITCODE -ne 0) {
        throw "Source I$source batch failed with exit code $LASTEXITCODE"
    }
}
