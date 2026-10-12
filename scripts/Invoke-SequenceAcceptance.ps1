[CmdletBinding()]
param(
    [switch]$AllowPaidApi,
    [ValidateRange(1,30)][int]$MaxApiCalls=30,
    [Parameter(Mandatory)][string]$BudgetFile
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if (-not $AllowPaidApi) { throw 'This bundle includes paid API calls. Explicit authorization and -AllowPaidApi are required.' }
if (-not $env:OPENAI_API_KEY -or -not $env:OPENAI_MODEL) { throw 'Set OPENAI_API_KEY and OPENAI_MODEL for this process.' }
$budgetPath=[IO.Path]::GetFullPath($BudgetFile,$root)
if ((Split-Path -Parent $budgetPath) -ne (Join-Path $root 'results')) { throw 'BudgetFile must be directly inside project results.' }
if (Test-Path -LiteralPath $budgetPath) { throw 'Budget already used. Inspect results; this command does not retry or reset authorization.' }
Push-Location -LiteralPath $root
$oldPythonPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=Join-Path $root 'src'
    & python -m unittest discover -s tests -q
    if ($LASTEXITCODE -ne 0) { throw 'Windows tests failed; no paid calls made.' }
    $linuxRoot=(& wsl.exe -d Ubuntu-24.04 -- wslpath -a $root.Replace('\','/')).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'WSL path resolution failed.' }
    & wsl.exe -d Ubuntu-24.04 --cd $linuxRoot -- "$linuxRoot/.runtime/bin/micromamba" run -p "$linuxRoot/.runtime/steve1" python -m unittest discover -s tests -q
    if ($LASTEXITCODE -ne 0) { throw 'WSL tests failed; no paid calls made.' }
    & python scripts/run_sequence_acceptance.py --allow-paid-api --max-api-calls $MaxApiCalls --budget-file $budgetPath
    if ($LASTEXITCODE -ne 0) { throw 'Integrated acceptance failed. Inspect preserved evidence; no automatic retry.' }
} finally {
    $env:PYTHONPATH=$oldPythonPath
    Pop-Location
}
