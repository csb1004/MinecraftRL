<#
CPU regression checks only. Uses existing isolated runtimes; never installs,
downloads, launches a game, trains a policy, or removes results/STOP.
#>
[CmdletBinding()]
param(
    [ValidateSet('All', 'Unit', 'Host')][string]$Profile = 'All',
    [string]$Distribution = 'Ubuntu-24.04',
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$resultsRoot = [IO.Path]::GetFullPath((Join-Path $root 'results'))
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $resultsRoot ('checks-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ'))
}
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $output.StartsWith($resultsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Check output must be a fresh directory under this project results folder.'
}
if (Test-Path -LiteralPath $output) { throw 'Check output already exists; choose a fresh directory.' }
# Reject junctions/symlinks in existing ancestors before creating output.
$ancestor = Split-Path -Parent $output
while ($ancestor -and $ancestor.Length -ge $root.Length) {
    if (Test-Path -LiteralPath $ancestor) {
        if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Check output cannot traverse a junction or symbolic link.'
        }
    }
    $ancestor = Split-Path -Parent $ancestor
}
New-Item -ItemType Directory -Path $output | Out-Null
$checks = [System.Collections.Generic.List[object]]::new()
$stop = Join-Path $root 'results/STOP'
$stopBefore = if (Test-Path -LiteralPath $stop) { (Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash } else { $null }
$originalEnvironment = @{}
foreach ($name in @('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES')) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$failure = $null
Push-Location -LiteralPath $root
try {
    $syntaxErrors = @()
    $powershellFiles = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1' -Recurse -File)
    foreach ($file in $powershellFiles) {
        $tokens = $null; $parseErrors = $null
        [void][Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$parseErrors)
        foreach ($errorItem in $parseErrors) {
            $syntaxErrors += [pscustomobject]@{file=$file.Name; message=$errorItem.Message; line=$errorItem.Extent.StartLineNumber}
        }
    }
    $checks.Add([pscustomobject]@{name='powershell_syntax'; files=$powershellFiles.Count; exit_code=[int]($syntaxErrors.Count -gt 0); errors=$syntaxErrors})
    if ($syntaxErrors.Count) { throw 'PowerShell syntax errors; see summary.json.' }

    if ($Profile -in @('All', 'Unit')) {
        Write-Output 'Running CPU unit checks in the existing WSL policy environment.'
        $resolved = & wsl.exe -d $Distribution -- wslpath -a $root.Replace('\', '/')
        if ($LASTEXITCODE -ne 0 -or -not $resolved) { throw 'Cannot resolve this project in WSL.' }
        $linux = ([string]$resolved).Trim()
        $arguments = @('-d', $Distribution, '--', 'env', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib',
            'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1',
            'OMP_NUM_THREADS=2', "PYTHONPATH=$linux/src:$linux/tests",
            "$linux/.runtime/bin/micromamba", 'run', '-p', "$linux/.runtime/steve1", 'python',
            '-m', 'unittest', 'discover', '-s', "$linux/tests", '-v')
        & wsl.exe @arguments *> (Join-Path $output 'unit-wsl.log')
        $checks.Add([pscustomobject]@{name='unit_wsl_cpu'; exit_code=$LASTEXITCODE; log='unit-wsl.log'})
        Get-Content -LiteralPath (Join-Path $output 'unit-wsl.log') -Tail 4
    }
    if ($Profile -in @('All', 'Host')) {
        Write-Output 'Running Windows PowerShell and offline world-fixture checks.'
        $python = Join-Path $root '.runtime/fixture-tools/Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $python)) { throw 'The existing isolated fixture-tools Python is required for Host checks.' }
        $env:PYTHONPATH = (Join-Path $root 'src') + ';' + (Join-Path $root 'tests')
        $env:PYTHONDONTWRITEBYTECODE = '1'
        $env:CUDA_VISIBLE_DEVICES = ''
        & $python -m unittest test_artifact_powershell test_priority_command_script test_rl_fixture test_startup_control -v *> (Join-Path $output 'host.log')
        $checks.Add([pscustomobject]@{name='host_fixtures'; exit_code=$LASTEXITCODE; log='host.log'})
        Get-Content -LiteralPath (Join-Path $output 'host.log') -Tail 4
    }
} catch {
    $failure = $_.Exception.Message
} finally {
    foreach ($name in $originalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], 'Process')
    }
    Pop-Location
    $stopAfter = if (Test-Path -LiteralPath $stop) { (Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash } else { $null }
    $stopUnchanged = $stopBefore -eq $stopAfter
    $passed = (-not $failure) -and $stopUnchanged -and -not @($checks | Where-Object {$_.exit_code -ne 0}).Count
    [ordered]@{
        schema='project_checks_v1'; utc=[DateTime]::UtcNow.ToString('o'); profile=$Profile
        passed=$passed; checks=@($checks.ToArray()); error=$failure; stop_unchanged=$stopUnchanged
        scope='CPU regression and host fixtures only; no real policy performance validation'
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'summary.json') -Encoding utf8
}
Write-Output "Check report: $output"
if (-not $passed) { throw 'Checks failed; inspect summary.json and logs. No experiment was started.' }
