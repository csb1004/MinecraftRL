[CmdletBinding()]
param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$batch=Join-Path $root 'results/rm-purpose-conditioning-20261031'
$stop=Join-Path $root 'results/STOP'
if (Test-Path -LiteralPath $stop) { throw 'STOP is latched; explicit re-arm is required.' }
if (-not (Test-Path -LiteralPath (Join-Path $batch 'goal-bank/manifest.json'))) { throw 'Prepared P2 bank required' }
$linuxRoot=(& wsl.exe -d $Distribution -- wslpath -a $root.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'WSL mapping failed' }
$seedList=@(81031001,81031003,81031007,81031009,81031013)
$modes=@('official','purpose-fixed','purpose-switch')
$plan=@()
for ($i=0; $i -lt $seedList.Count; $i++) {
    for ($j=0; $j -lt 3; $j++) {
        $mode=$modes[($i+$j)%3]
        $plan+=@{seed=$seedList[$i];mode=$mode;name=([string]$seedList[$i]+'-'+$mode)}
    }
}
$planPath=Join-Path $batch 'pilot-plan.json'
if (Test-Path -LiteralPath $planPath) { throw 'P2 already started; inspect existing evidence. No automatic rerun.' }
$protocol=Join-Path $root 'docs/specs/2026-10-31-purpose-conditioning-p2.md'
$bankHash=(Get-FileHash -LiteralPath (Join-Path $batch 'goal-bank/manifest.json') -Algorithm SHA256).Hash
@{schema=1;protocol='P2';seeds=$seedList;runs=$plan;limit_s=180;split='development';optimizer_updates=0;
    bank_sha256=$bankHash;protocol_sha256=(Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding utf8
$snapshot=Join-Path $batch 'source-at-game'
New-Item -ItemType Directory -Path $snapshot -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'src') -Destination $snapshot -Recurse
Copy-Item -LiteralPath (Join-Path $root 'configs/experiment.json') -Destination $snapshot
Copy-Item -LiteralPath $protocol -Destination $snapshot
Copy-Item -LiteralPath $PSCommandPath -Destination $snapshot
Copy-Item -LiteralPath (Join-Path $root 'scripts/prepare_rm_purpose_bank.py') -Destination $snapshot
$sourceHashes=Get-ChildItem -LiteralPath (Join-Path $root 'src/minecraftrl') -File -Filter '*.py' |
    ForEach-Object { @{path=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} }
@{git_head=(git -C $root rev-parse HEAD);source=$sourceHashes;protocol_sha256=(Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash;
    modified_worktree=$true;source_snapshot='source-at-game'} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
foreach ($run in $plan) {
    if (Test-Path -LiteralPath $stop) { Write-Output 'STOP observed; remaining conditions unexecuted.'; break }
    foreach ($source in $sourceHashes) {
        if ((Get-FileHash -LiteralPath (Join-Path $root ('src/minecraftrl/'+$source.path)) -Algorithm SHA256).Hash -ne $source.sha256) {
            throw 'Source changed during locked P2 batch'
        }
    }
    if ((Get-FileHash -LiteralPath (Join-Path $batch 'goal-bank/manifest.json') -Algorithm SHA256).Hash -ne $bankHash) {
        throw 'Goal bank manifest changed'
    }
    $argv=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',
        "PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache",
        "$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python','-m','minecraftrl.cli',
        'episode','--root',$linuxRoot,'--policy','steve1','--condition','wood','--seed',([string]$run.seed),
        '--limit-s','180','--rm-conditioning',$run.mode,'--rm-goal-bank',"$linuxRoot/results/rm-purpose-conditioning-20261031/goal-bank",
        '--output',"$linuxRoot/results/rm-purpose-conditioning-20261031/$($run.name)",'--stop-file',"$linuxRoot/results/STOP")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($run.name+'-command.json')) -Encoding utf8
    Write-Output ('Starting '+$run.name)
    & wsl.exe @argv *> (Join-Path $batch ($run.name+'-console.log'))
    $runExit=$LASTEXITCODE
    $summaryPath=Join-Path $batch ($run.name+'/summary.json')
    if (-not (Test-Path -LiteralPath $summaryPath)) { throw ('No supervisor summary: '+$run.name+' exit '+$runExit) }
    $summary=Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    Write-Output ($run.name+': '+$summary.status+' steps='+$summary.steps+' server_goal='+$summary.server_target_reached)
    if ($runExit -ne 0 -or -not $summary.execution_valid) {
        Write-Output 'Technical error/cancellation; preserve evidence. Remaining runs unexecuted.'
        break
    }
}
Write-Output 'P2 driver ended. Audit actual dispatches and server inventory before interpreting performance.'
