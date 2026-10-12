[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SessionPath,
    [switch]$AllowPaidApi,
    [ValidateRange(1,20)][int]$MaxApiCalls=5,
    [string]$MemoryFile,
    [string]$UserId,
    [string]$PersonaFile,
    [switch]$InspectMemory
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$previousPythonPath=$env:PYTHONPATH
$previousEncoding=$env:PYTHONIOENCODING
try {
    $env:PYTHONPATH=Join-Path $root 'src'
    $env:PYTHONIOENCODING='utf-8'
    $arguments=@('-m','minecraftrl.dialogue_console','--session',$SessionPath,'--max-api-calls',"$MaxApiCalls")
    if ($AllowPaidApi) { $arguments+='--allow-paid-api' }
    if ($InspectMemory) { $arguments+='--inspect-memory' }
    if ($UserId) { $arguments+=@('--user-id',$UserId) }
    if ($MemoryFile) {
        if (-not $UserId) { throw 'MemoryFile requires an explicitly selected UserId.' }
        $arguments+=@('--memory-file',$MemoryFile)
    }
    if ($PersonaFile) { $arguments+=@('--persona-file',$PersonaFile) }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Dialogue console returned $LASTEXITCODE." }
} finally {
    $env:PYTHONPATH=$previousPythonPath
    $env:PYTHONIOENCODING=$previousEncoding
}
