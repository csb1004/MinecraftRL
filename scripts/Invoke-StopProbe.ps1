[CmdletBinding()]
param(
    [ValidateSet('vpt','steve1')][string]$Policy = 'vpt',
    [Parameter(Mandatory)][ValidateSet('initialization','inference','step')][string]$Phase,
    [ValidateRange(1,1000)][int]$TriggerStep = 1,
    [string]$Distribution = 'Ubuntu-24.04'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $root 'results/STOP')) {
    throw 'STOP is latched. Inspect the previous shutdown before re-arming.'
}
$resolved = & wsl.exe -d $Distribution -- wslpath -a $root.Replace('\','/')
if ($LASTEXITCODE -ne 0 -or -not $resolved) { throw 'Cannot resolve project path in WSL.' }
$linuxRoot = ([string]$resolved).Trim()
# Environment variables apply only to this diagnostic process tree.
$probeArgs = @('-d', $Distribution, '--', 'env',
    "MAMBA_ROOT_PREFIX=$linuxRoot/.runtime/mamba",
    "CONDA_PKGS_DIRS=$linuxRoot/.runtime/mamba/pkgs",
    "XDG_CACHE_HOME=$linuxRoot/.cache",
    "HF_HOME=$linuxRoot/.cache/huggingface",
    "CUDA_CACHE_PATH=$linuxRoot/.cache/cuda",
    'GALLIUM_DRIVER=d3d12', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib',
    "$linuxRoot/.runtime/bin/micromamba", 'run', '-p', "$linuxRoot/.runtime/$Policy",
    'python', "$linuxRoot/scripts/run_stop_probe.py", '--policy', $Policy, '--phase', $Phase,
    '--trigger-step', "$TriggerStep")
& wsl.exe @probeArgs
if ($LASTEXITCODE -ne 0) { throw "STOP probe returned $LASTEXITCODE. Inspect its result directory." }
Write-Output 'Inspect stop-test-evaluation.json; this command intentionally leaves results/STOP latched.'
