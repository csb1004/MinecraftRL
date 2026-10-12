[CmdletBinding()]
param()
$root = Split-Path -Parent $PSScriptRoot
# All probe launches from wsl-probe.sh use this one project-scoped stop file.
New-Item -ItemType File -Path (Join-Path $root 'results\STOP') -Force | Out-Null
Write-Host 'Stop requested. Inspect stop_acknowledged/input_release and summary.json; an in-flight step may finish.'
