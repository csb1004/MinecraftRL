[CmdletBinding()]
param(
    [ValidateSet('vpt','steve1')][string]$Policy = 'vpt',
    [ValidateSet('setup','resume-setup','download','preflight','episode','suite','session','task-scenario','dialogue-scenario','world-scenario','semantic-world-scenario')][string]$Operation = 'preflight',
    [ValidateSet('spontaneous','wood','dirt')][string]$Condition,
    [string]$Distribution = 'Ubuntu-24.04',
    [int]$Seed = 104729,
    [ValidateRange(1,180)][int]$Seconds = 180,
    [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$WorldSlot
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# Forward slashes survive wsl.exe's argument forwarding without shell escapes.
$resolved = & wsl.exe -d $Distribution -- wslpath -a $root.Replace('\','/')
if ($LASTEXITCODE -ne 0 -or -not $resolved) { throw 'Cannot resolve project path in WSL.' }
$linuxRoot = ([string]$resolved).Trim()
# Arguments are passed as separate tokens; no interpolated bash -c commands.
$argsForWsl = @('-d', $Distribution, '--', 'bash', "$linuxRoot/scripts/wsl-probe.sh", $linuxRoot, $Policy, $Operation, "$Seed", "$Seconds")
if ($WorldSlot -and $Operation -ne 'session') { throw 'WorldSlot requires Operation session.' }
if ($Condition) { $argsForWsl += $Condition }
elseif ($WorldSlot) { $argsForWsl += 'wood' }
if ($WorldSlot) { $argsForWsl += $WorldSlot }
& wsl.exe @argsForWsl
if ($LASTEXITCODE -ne 0) { throw "Policy probe returned $LASTEXITCODE. Inspect results; no API fallback was used." }
