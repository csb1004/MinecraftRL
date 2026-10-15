#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')][string]$Repository='csb1004/MinecraftRL',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$Tag='research-20261015',
    [string]$Experiment='oct15-motion-corner',
    [string]$ProjectRoot=(Split-Path $PSScriptRoot -Parent)
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'GitHub-ArtifactHelpers.ps1')
$root=(Resolve-Path -LiteralPath $ProjectRoot).Path
$cache=Get-ArtifactCache -Root $root
$headers=Get-ArtifactHeaders
$release=Get-ArtifactRelease -Repository $Repository -Tag $Tag -Headers $headers
if (-not $release -or $release.draft) { throw 'Published release not found.' }
$assets=@(Get-ArtifactAssets -Repository $Repository -ReleaseId $release.id -Headers $headers)
$metadata=@($assets | Where-Object name -CEQ 'artifact-manifest.json')
if ($metadata.Count -ne 1 -or $metadata[0].digest -notmatch '^sha256:([0-9a-f]{64})$') { throw 'Manifest SHA256 unavailable.' }
$manifestSha=$matches[1]
$manifestPath=Join-Path $cache ($manifestSha+'.json')
Save-ArtifactAsset -Asset $metadata[0] -Path $manifestPath -Sha256 $manifestSha -Bytes $metadata[0].size
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.repository -cne $Repository -or $manifest.tag -cne $Tag) { throw 'Release manifest identity mismatch.' }
$tagCommit=Get-ArtifactTagCommit -Repository $Repository -Tag $Tag -Headers $headers
if ($tagCommit -cne $manifest.code_commit) { throw 'Manifest code commit does not match the release tag.' }
$tool=Join-Path $PSScriptRoot '../src/minecraftrl/artifact_release.py'
$planText=& python $tool plan --root $root --manifest $manifestPath --experiment $Experiment
if ($LASTEXITCODE -ne 0) { throw 'Artifact conflict or invalid manifest. No project artifact was overwritten.' }
$plan=$planText | ConvertFrom-Json
$downloads=0
foreach ($group in $plan.groups) {
    $sourceRelease=Get-ArtifactRelease -Repository $Repository -Tag $group.release_tag -Headers $headers
    if (-not $sourceRelease -or $sourceRelease.draft) { throw 'Referenced release unavailable.' }
    $sourceAssets=@(Get-ArtifactAssets -Repository $Repository -ReleaseId $sourceRelease.id -Headers $headers)
    $asset=@($sourceAssets | Where-Object name -CEQ $group.asset)
    if ($asset.Count -ne 1) { throw 'Referenced asset unavailable.' }
    $destination=Join-Path $cache $group.asset
    $hadCache=Test-Path -LiteralPath $destination
    Save-ArtifactAsset -Asset $asset[0] -Path $destination -Sha256 $group.sha256 -Bytes $group.bytes
    if (-not $hadCache) { $downloads++ }
}
$result=& python $tool restore --root $root --manifest $manifestPath --experiment $Experiment --cache $cache
if ($LASTEXITCODE -ne 0) { throw 'Artifact restoration failed. Inspect conflicts or hashes.' }
$result
Write-Host "Downloaded bundles: $downloads; release code commit: $($manifest.code_commit)"
Write-Host 'Artifacts verified. Base models and Minecraft runtime require the documented official/local setup. No game was launched.'
