param([Parameter(Mandatory=$true)][string]$LiteralPath)
$ErrorActionPreference='Stop'
$projectRoot=[System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$file=Get-Item -LiteralPath $LiteralPath
if ($file.PSIsContainer -or -not $file.FullName.StartsWith($projectRoot+'\',[System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Artifact must be a file within this project.'
}
# A fresh full read on every call. Read sharing forbids concurrent write/delete.
$stream=[System.IO.File]::Open($file.FullName,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::Read)
$algorithm=[System.Security.Cryptography.SHA256]::Create()
try {
    $length=$stream.Length
    $digest=([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant()
    @{schema=1;sha256=$digest;bytes=$length} | ConvertTo-Json -Compress
} finally {
    $algorithm.Dispose()
    $stream.Dispose()
}
