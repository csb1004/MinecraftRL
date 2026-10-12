[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$WorldSlot='companion',
    [string]$UserId='local-operator',
    [string]$Distribution='Ubuntu-24.04',
    [int]$Seed=104729,
    [ValidateRange(1,180)][int]$Seconds=180,
    [switch]$AllowPaidApi,
    [ValidateRange(1,20)][int]$MaxApiCalls=5
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$previousPythonPath=$env:PYTHONPATH
$previousEncoding=$env:PYTHONIOENCODING
try {
    $env:PYTHONPATH=Join-Path $root 'src'
    $env:PYTHONIOENCODING='utf-8'
    $arguments=@('-m','minecraftrl.companion','--world-slot',$WorldSlot,'--user-id',$UserId,
        '--distribution',$Distribution,'--seed',"$Seed",'--seconds',"$Seconds",'--max-api-calls',"$MaxApiCalls")
    if ($AllowPaidApi) { $arguments+='--allow-paid-api' }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Session ended with an unverified result ($LASTEXITCODE). Inspect the printed result folder." }
} finally {
    $env:PYTHONPATH=$previousPythonPath
    $env:PYTHONIOENCODING=$previousEncoding
}
