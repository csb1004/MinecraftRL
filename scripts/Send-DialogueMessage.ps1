[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SessionPath,
    [Parameter(Mandatory)][ValidateLength(1,2048)][string]$Text,
    [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$TaskId='gather',
    [ValidateRange(1,2147483647)][int]$ExpectedRevision,
    [ValidatePattern('^[A-Za-z0-9_-]{1,64}$')][string]$MessageId=([guid]::NewGuid().ToString('N'))
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$resolved=[IO.Path]::GetFullPath($SessionPath)
$boundary=[IO.Path]::GetFullPath((Join-Path $root 'results'))+[IO.Path]::DirectorySeparatorChar
if (-not $resolved.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $resolved -PathType Container)) {
    throw 'SessionPath must be an existing session directory inside project results.'
}
# Reject any linked ancestor down to results, not just the leaf directory.
$directory=Get-Item -LiteralPath $resolved
while ($directory.FullName.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or $directory.FullName -eq $boundary.TrimEnd([IO.Path]::DirectorySeparatorChar)) {
    if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked session path rejected.' }
    $directory=$directory.Parent
}
if (Test-Path -LiteralPath (Join-Path $resolved 'summary.json')) { throw 'This session has ended.' }
if (Test-Path -LiteralPath (Join-Path $root 'results/STOP')) { throw 'Global STOP is latched.' }
$commands=Join-Path $resolved 'commands'
if (Test-Path -LiteralPath $commands) {
    if ((Get-Item -LiteralPath $commands).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked command directory rejected.' }
} else { [void](New-Item -ItemType Directory -Path $commands) }
$message=@{ kind='dialogue'; message_id=$MessageId; text=$Text; task_id=$TaskId }
if ($PSBoundParameters.ContainsKey('ExpectedRevision')) { $message.expected_revision=$ExpectedRevision }
$nanoseconds=([DateTimeOffset]::UtcNow.Ticks-[DateTimeOffset]::UnixEpoch.Ticks)*[long]100
$name=$nanoseconds.ToString('D20')+'-'+[guid]::NewGuid().ToString('N')
$temporary=Join-Path $commands ($name+'.tmp'); $final=Join-Path $commands ($name+'.json')
[IO.File]::WriteAllText($temporary,($message | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
Move-Item -LiteralPath $temporary -Destination $final
@{message_id=$MessageId; file=$final; status='submitted; inspect dialogue_decision and command_acknowledged in events.jsonl'} | ConvertTo-Json
