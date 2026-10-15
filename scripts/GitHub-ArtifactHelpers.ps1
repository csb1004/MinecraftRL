Set-StrictMode -Version Latest

function Get-ArtifactCache {
    param([string]$Root)
    $path=$Root
    foreach ($part in @('.cache','release-artifacts')) {
        $path=Join-Path $path $part
        if (Test-Path -LiteralPath $path) {
            $item=Get-Item -LiteralPath $path -Force
            if (-not $item.PSIsContainer -or ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw 'Artifact cache must be an ordinary project-local directory.' }
        } else { New-Item -ItemType Directory -Path $path | Out-Null }
    }
    return $path
}

function Get-ArtifactTagCommit {
    param([string]$Repository,[string]$Tag,[hashtable]$Headers)
    $reference=Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/git/ref/tags/$Tag" -Headers $Headers
    $object=$reference.object
    for ($depth=0; $depth -lt 5; $depth++) {
        if ($object.type -eq 'commit') { return $object.sha }
        if ($object.type -ne 'tag') { break }
        $annotated=Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/git/tags/$($object.sha)" -Headers $Headers
        $object=$annotated.object
    }
    throw 'Release tag does not resolve to a commit.'
}

function Get-ArtifactHeaders {
    param([switch]$Authenticated)
    $headers=@{'User-Agent'='MinecraftRL-artifacts';'Accept'='application/vnd.github+json';'X-GitHub-Api-Version'='2026-03-10'}
    if ($Authenticated) {
        $token=$env:GITHUB_TOKEN
        if (-not $token) {
            $lines="protocol=https`nhost=github.com`n`n" | git -c credential.interactive=never credential fill
            if ($LASTEXITCODE -ne 0) { throw 'Existing GitHub credential unavailable.' }
            foreach ($line in $lines) { if ($line.StartsWith('password=')) { $token=$line.Substring(9) } }
            $lines=$null
        }
        if (-not $token) { throw 'No existing GitHub credential. Configure Git authentication before publishing.' }
        $headers.Authorization='Bearer '+$token
        $token=$null
    }
    return $headers
}

function Get-ArtifactRelease {
    param([string]$Repository,[string]$Tag,[hashtable]$Headers)
    # Listing also finds drafts, so a failed upload can resume without a second release.
    for ($page=1; ; $page++) {
        $items=@(Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/releases?per_page=100&page=$page" -Headers $Headers)
        foreach ($item in $items) { if ($item.tag_name -ceq $Tag) { return $item } }
        if ($items.Count -lt 100) { return $null }
    }
}

function Get-ArtifactAssets {
    param([string]$Repository,[long]$ReleaseId,[hashtable]$Headers)
    for ($page=1; ; $page++) {
        $items=@(Invoke-RestMethod -Uri "https://api.github.com/repos/$Repository/releases/$ReleaseId/assets?per_page=100&page=$page" -Headers $Headers)
        foreach ($item in $items) { Write-Output $item }
        if ($items.Count -lt 100) { break }
    }
}

function Assert-ArtifactAsset {
    param($Asset,[string]$Sha256,[long]$Bytes)
    if (-not $Asset -or $Asset.state -ne 'uploaded' -or $Asset.size -ne $Bytes -or $Asset.digest -cne "sha256:$Sha256") {
        throw 'GitHub asset state, size, or SHA256 does not match the manifest.'
    }
}

function Save-ArtifactAsset {
    param($Asset,[string]$Path,[string]$Sha256,[long]$Bytes)
    Assert-ArtifactAsset -Asset $Asset -Sha256 $Sha256 -Bytes $Bytes
    if (Test-Path -LiteralPath $Path) {
        if ((Get-Item -LiteralPath $Path).Length -eq $Bytes -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ceq $Sha256) { return }
        throw "Conflicting cached file: $Path"
    }
    $url=[Uri]$Asset.browser_download_url
    if ($url.Scheme -ne 'https' -or $url.Host -ne 'github.com') { throw 'Unexpected asset download host.' }
    $partial=$Path+'.'+[Guid]::NewGuid().ToString('N')+'.partial'
    try {
        # Public downloads deliberately receive no GitHub credential header.
        Invoke-WebRequest -Uri $url.AbsoluteUri -OutFile $partial -TimeoutSec 900 -Headers @{'User-Agent'='MinecraftRL-artifacts'}
        if ((Get-Item -LiteralPath $partial).Length -ne $Bytes -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Sha256) { throw 'Downloaded asset hash mismatch.' }
        [System.IO.File]::Move($partial,$Path)
    } finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    }
}
