[CmdletBinding()]
param([ValidateSet('vpt','steve1')][string]$Policy = 'vpt')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$lock = Get-Content -LiteralPath (Join-Path $root 'configs\sources.lock.json') -Raw | ConvertFrom-Json
$names = @($Policy, 'minerl', 'gym')
if ($Policy -eq 'steve1') { $names += @('mineclip', 'minedojo') }
$external = Join-Path $root 'external'
New-Item -ItemType Directory -Path $external -Force | Out-Null
foreach ($name in $names) {
    $spec = $lock.sources.$name
    $dest = Join-Path $external $name
    if (Test-Path -LiteralPath $dest) {
        $actual = & git -C $dest rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $actual -ne $spec.commit) { throw "Existing source mismatch: $dest" }
        & git -C $dest diff --quiet HEAD --
        if ($LASTEXITCODE -ne 0) { throw "Existing source has tracked edits: $dest" }
        continue
    }
    & git init $dest
    if ($LASTEXITCODE -ne 0) { throw "git init failed: $name" }
    & git -C $dest remote add origin $spec.repository
    if ($LASTEXITCODE -ne 0) { throw "git remote failed: $name" }
    & git -C $dest -c core.autocrlf=false fetch --depth 1 origin $spec.commit
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed: $name" }
    & git -C $dest -c core.autocrlf=false checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed: $name" }
}
Write-Host 'Pinned source checkouts ready. Runtime compatibility is not yet validated.'
