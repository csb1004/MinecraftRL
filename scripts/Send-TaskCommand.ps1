[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SessionPath,
    [Parameter(Mandatory)][ValidateSet('request','modify','request_plan','modify_plan','pause','resume','cancel')][string]$Operation,
    [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$TaskId='gather',
    [ValidateRange(1,2147483647)][int]$ExpectedRevision=1,
    [ValidateSet('wood','dirt')][string]$Goal='wood',
    [ValidateRange(1,64)][int]$Count=4,
    [string]$StepsJson
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$resolved=[IO.Path]::GetFullPath($SessionPath)
$boundary=[IO.Path]::GetFullPath((Join-Path $root 'results'))+[IO.Path]::DirectorySeparatorChar
if (-not $resolved.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $resolved -PathType Container)) {
    throw 'SessionPath must be an existing session directory inside project results.'
}
if ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked session directory rejected.' }
if (Test-Path -LiteralPath (Join-Path $resolved 'summary.json')) { throw 'This session has ended.' }
if (Test-Path -LiteralPath (Join-Path $root 'results/STOP')) { throw 'Global STOP is latched.' }
$commands=Join-Path $resolved 'commands'
if (Test-Path -LiteralPath $commands) {
    if ((Get-Item -LiteralPath $commands).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked command directory rejected.' }
} else { [void](New-Item -ItemType Directory -Path $commands) }
$id=[guid]::NewGuid().ToString('N')
$command=@{ command_id=$id; op=$Operation; task_id=$TaskId }
if ($Operation -notin @('request','request_plan')) { $command.expected_revision=$ExpectedRevision }
if ($Operation -in @('request','modify')) { $command.goal=$Goal; $command.count=$Count }
if ($Operation -in @('request_plan','modify_plan')) {
    $steps=ConvertFrom-Json -InputObject $StepsJson -NoEnumerate
    if ($steps -isnot [array] -or $steps.Count -lt 1 -or $steps.Count -gt 8) { throw 'StepsJson requires 1 to 8 stages.' }
    foreach ($stage in $steps) {
        if (($stage.PSObject.Properties.Name | Sort-Object) -join ',' -ne 'count,goal' -or $stage.goal -notin @('wood','dirt') -or $stage.count -isnot [long] -and $stage.count -isnot [int] -or $stage.count -lt 1 -or $stage.count -gt 64) { throw 'Invalid collection stage.' }
    }
    $command.steps=$steps
}
# Match Python time_ns filename ordering using UTC Unix ticks, then a unique id.
$nanoseconds=([DateTimeOffset]::UtcNow.Ticks-[DateTimeOffset]::UnixEpoch.Ticks)*[long]100
$name=$nanoseconds.ToString('D20')+'-'+$id
$temporary=Join-Path $commands ($name+'.tmp'); $final=Join-Path $commands ($name+'.json')
[IO.File]::WriteAllText($temporary,($command | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporary -Destination $final
@{command_id=$id; file=$final; status='submitted; inspect command_acknowledged or command_rejected in events.jsonl'} | ConvertTo-Json
