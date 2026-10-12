[CmdletBinding()]
param([string]$Distribution = 'Ubuntu-24.04')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root ('results\windows-preflight-' + (Get-Date -Format 'yyyyMMddTHHmmss'))
New-Item -ItemType Directory -Path $output | Out-Null
$report = [ordered]@{ timestamp = (Get-Date).ToUniversalTime().ToString('o'); project = $root; os = [Environment]::OSVersion.ToString(); powershell = $PSVersionTable.PSVersion.ToString(); commands = @{}; api_key_present = [bool]$env:OPENAI_API_KEY }
# Report presence only; never print environment variables or credential values.
$checks = @{
    'gpu' = @('nvidia-smi.exe', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader')
    'python' = @('python.exe', '--version')
    'node' = @('node.exe', '--version')
    'java' = @('java.exe', '-version')
    'git' = @('git.exe', '--version')
    'wsl' = @('wsl.exe', '--list', '--verbose')
}
foreach ($name in $checks.Keys) {
    $parts = $checks[$name]; $exe = Get-Command $parts[0] -ErrorAction SilentlyContinue
    if (-not $exe) { $report.commands[$name] = @{ available = $false }; continue }
    try {
        $values = @(& $parts[0] $parts[1..($parts.Length - 1)] 2>&1 | ForEach-Object { "$_" })
        $report.commands[$name] = @{ available = $true; exit_code = $LASTEXITCODE; output = $values }
    } catch { $report.commands[$name] = @{ error = $_.Exception.Message } }
}
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    $report['wsl_gpu'] = @(& wsl.exe -d $Distribution -- nvidia-smi 2>&1 | ForEach-Object { "$_" })
    $report['wsl_gl'] = @(& wsl.exe -d $Distribution -- bash -lc 'command -v glxinfo && glxinfo -B' 2>&1 | ForEach-Object { "$_" })
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'preflight.json') -Encoding UTF8
Write-Host "Read-only report saved: $output"
