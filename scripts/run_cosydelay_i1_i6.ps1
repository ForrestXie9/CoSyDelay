param(
    [int[]]$Intersections = (1..6),
    [int]$SeedBase = 20260712,
    [string]$OutputRoot = "runs\cosydelay_i1_i6"
)

$ErrorActionPreference = "Stop"
$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $PackageRoot
$env:PYTHONPATH = (Resolve-Path .\code).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($intersection in $Intersections) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $output = Join-Path $OutputRoot ("I{0}_{1}" -f $intersection, $stamp)
    Write-Host "Starting CoSyDelay intersection $intersection -> $output"
    & python -u -m methods.cosydelay.run_p10g10_100 `
        --intersection $intersection `
        --data-dir .\data\locked_splits `
        --seed-base ($SeedBase + $intersection * 100000) `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "CoSyDelay intersection $intersection failed with exit code $LASTEXITCODE"
    }
}
