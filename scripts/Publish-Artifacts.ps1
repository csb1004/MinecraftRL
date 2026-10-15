#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BundleDirectory,
    [Parameter(Mandatory)][string]$NotesFile,
    [switch]$DraftOnly
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'GitHub-ArtifactHelpers.ps1')
$root=Split-Path $PSScriptRoot -Parent
$directory=(Resolve-Path -LiteralPath $BundleDirectory).Path
$manifestPath=Join-Path $directory 'artifact-manifest.json'
$manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
# Validation and local conflict checking happen before any network mutation.
$validation=& python (Join-Path $root 'src/minecraftrl/artifact_release.py') plan --root $root --manifest $manifestPath --experiment all
if ($LASTEXITCODE -ne 0) { throw 'Artifact manifest validation failed.' }
$head=(& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -cne $manifest.code_commit) { throw 'Manifest must identify the current code commit.' }
if (@(& git -C $root status --porcelain --untracked-files=normal).Count) { throw 'Commit reviewed source changes before publishing.' }
foreach ($group in $manifest.groups) {
    if ($group.release_tag -ceq $manifest.tag) {
        $file=Join-Path $directory $group.asset
        if ((Get-Item -LiteralPath $file).Length -ne $group.bytes -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -cne $group.sha256) { throw 'Local bundle hash mismatch.' }
    }
}
$notes=Get-Content -LiteralPath $NotesFile -Raw
$headers=Get-ArtifactHeaders -Authenticated
try {
    $repository=$manifest.repository
    $repo=Invoke-RestMethod -Uri "https://api.github.com/repos/$repository" -Headers $headers
    if ($repo.private -or -not $repo.permissions.push) { throw 'This publisher requires the approved public repository and push access.' }
    $commit=Invoke-RestMethod -Uri "https://api.github.com/repos/$repository/commits/$head" -Headers $headers
    if ($commit.sha -cne $head) { throw 'Push the exact code commit before publishing.' }
    # Resolve earlier shared bundles before creating or publishing this release.
    foreach ($group in $manifest.groups) {
        if ($group.release_tag -cne $manifest.tag) {
            $old=Get-ArtifactRelease -Repository $repository -Tag $group.release_tag -Headers $headers
            if (-not $old -or $old.draft) { throw 'Referenced release must already be published.' }
            $assets=@(Get-ArtifactAssets -Repository $repository -ReleaseId $old.id -Headers $headers)
            $asset=@($assets | Where-Object name -CEQ $group.asset)
            if ($asset.Count -ne 1) { throw 'Referenced asset missing or ambiguous.' }
            Assert-ArtifactAsset -Asset $asset[0] -Sha256 $group.sha256 -Bytes $group.bytes
        }
    }
    $release=Get-ArtifactRelease -Repository $repository -Tag $manifest.tag -Headers $headers
    if (-not $release) {
        # An existing tag is checked independently; target_commitish does not move tags.
        $refs=@(& git -C $root ls-remote origin "refs/tags/$($manifest.tag)" "refs/tags/$($manifest.tag)^{}")
        if ($LASTEXITCODE -ne 0) { throw 'Cannot verify remote release tag.' }
        if ($refs.Count -and (($refs[-1] -split '\s+')[0] -cne $head)) { throw 'Existing tag points to another commit.' }
        $body=@{tag_name=$manifest.tag;target_commitish=$head;name="MinecraftRL $($manifest.tag)";body=$notes;draft=$true;prerelease=$true;make_latest='false'} | ConvertTo-Json
        $release=Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$repository/releases" -Headers $headers -Body $body -ContentType 'application/json'
    }
    if ($release.target_commitish -cne $head) { throw 'Existing release is not bound to this exact commit.' }
    $uploads=@()
    foreach ($group in $manifest.groups) { if ($group.release_tag -ceq $manifest.tag) { $uploads+=@{name=$group.asset;path=(Join-Path $directory $group.asset);sha=$group.sha256;bytes=$group.bytes} } }
    $uploads+=@{name='artifact-manifest.json';path=$manifestPath;sha=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $manifestPath).Length}
    $existing=@(Get-ArtifactAssets -Repository $repository -ReleaseId $release.id -Headers $headers)
    foreach ($upload in $uploads) {
        $found=@($existing | Where-Object name -CEQ $upload.name)
        if ($found.Count -gt 1) { throw 'Duplicate GitHub asset names.' }
        if ($found.Count -eq 1) {
            Assert-ArtifactAsset -Asset $found[0] -Sha256 $upload.sha -Bytes $upload.bytes
            Write-Host "Verified existing asset: $($upload.name)"
            continue
        }
        if (-not $release.draft) { throw 'Published releases are not modified. Use a new release tag.' }
        Write-Host "Uploading: $($upload.name)"
        $url="https://uploads.github.com/repos/$repository/releases/$($release.id)/assets?name=$([Uri]::EscapeDataString($upload.name))"
        $uploaded=Invoke-RestMethod -Method Post -Uri $url -Headers $headers -InFile $upload.path -ContentType 'application/octet-stream' -TimeoutSec 900
        Assert-ArtifactAsset -Asset $uploaded -Sha256 $upload.sha -Bytes $upload.bytes
    }
    if ($release.draft -and -not $DraftOnly) {
        $release=Invoke-RestMethod -Method Patch -Uri "https://api.github.com/repos/$repository/releases/$($release.id)" -Headers $headers -Body (@{draft=$false;prerelease=$true;make_latest='false'} | ConvertTo-Json) -ContentType 'application/json'
    }
    $receipt=@{repository=$repository;tag=$manifest.tag;commit=$head;release_url=$release.html_url;draft=$release.draft;assets_verified=$uploads.Count}
    $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $directory 'publication-receipt.json') -Encoding utf8
    $receipt | ConvertTo-Json
} finally { $headers.Clear() }
