[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('vpt','steve1')][string]$Policy,
    [Parameter(Mandatory)][string]$WheelPath,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedSha256
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if ($projectRoot -ne 'D:\DATA\Desktop\MinecraftRL') { throw 'Unexpected project location' }
$wheel = (Resolve-Path -LiteralPath $WheelPath).Path
if (-not $wheel.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Wheel must be an artifact inside this project'
}
if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash -ne $ExpectedSha256) {
    throw 'Wheel SHA256 mismatch'
}
$pythonVersion = if ($Policy -eq 'vpt') { '3.9' } else { '3.10' }
$abi = if ($Policy -eq 'vpt') { 'cp39' } else { 'cp310' }
if ((Split-Path $wheel -Leaf) -ne "minerl-1.0.0-$abi-$abi-linux_x86_64.whl") {
    throw 'Unexpected MineRL wheel or Python ABI'
}
$site = Join-Path $projectRoot ".runtime/$Policy/lib/python$pythonVersion/site-packages"
if (-not (Test-Path -LiteralPath $site -PathType Container)) { throw 'Create the isolated environment first' }
foreach ($name in @('minerl','minerl-1.0.0.dist-info')) {
    if (Test-Path -LiteralPath (Join-Path $site $name)) { throw "Existing ${name}: inspect and preserve it before retrying" }
}
$installTemp = Join-Path $projectRoot '.runtime/windows-wheel-temp'
New-Item -ItemType Directory -Path $installTemp -Force | Out-Null
$prior = @{ TEMP=$env:TEMP; TMP=$env:TMP; PIP_CONFIG_FILE=$env:PIP_CONFIG_FILE }
try {
    $env:TEMP = $installTemp; $env:TMP = $installTemp; $env:PIP_CONFIG_FILE = 'NUL'
    # This only extracts a Linux wheel into its explicit target. It never runs
    # Linux code with Windows Python or installs into the Windows interpreter.
    & python -m pip install --no-index --no-deps --no-compile --no-cache-dir --disable-pip-version-check `
        --platform linux_x86_64 --implementation cp --python-version $pythonVersion --abi $abi --target $site $wheel
    if ($LASTEXITCODE -ne 0) { throw "Wheel file installation failed: $LASTEXITCODE" }
} finally {
    $env:TEMP=$prior.TEMP; $env:TMP=$prior.TMP; $env:PIP_CONFIG_FILE=$prior.PIP_CONFIG_FILE
}
Write-Host 'Files installed. Validate imports, package metadata and gameplay under WSL.'
