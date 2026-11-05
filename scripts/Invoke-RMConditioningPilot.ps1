[CmdletBinding()]
param([string]$Distribution='Ubuntu-24.04', [switch]$ResumeFinalizationFix, [switch]$ResumeStartupAfterIOCheck)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$batch=Join-Path $root 'results/rm-conditioning-20261031'
$stop=Join-Path $root 'results/STOP'
if (Test-Path -LiteralPath $stop) { throw 'STOP is latched; explicit re-arm is required.' }
$linuxRoot=(& wsl.exe -d $Distribution -- wslpath -a $root.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'WSL project mapping failed' }
$seedList=@(81031001,81031003,81031007,81031009,81031013)
$plan=@()
for ($i=0; $i -lt $seedList.Count; $i++) {
    $modes=if ($i % 2 -eq 0) { @('fixed','staged') } else { @('staged','fixed') }
    foreach ($mode in $modes) { $plan+=@{seed=$seedList[$i];mode=$mode;name=([string]$seedList[$i]+'-'+$mode)} }
}
$pendingPlan=$plan
$tag=if ($ResumeStartupAfterIOCheck) { '-startup-r1' } elseif ($ResumeFinalizationFix) { '-finalization-r1' } else { '' }
if ($ResumeFinalizationFix -and $ResumeStartupAfterIOCheck) { throw 'Choose one technical resume only' }
if ($ResumeStartupAfterIOCheck) {
    $diagnostic=Get-Content -LiteralPath (Join-Path $batch 'model-loading-diagnostic.json') -Raw | ConvertFrom-Json
    if ($diagnostic.status -ne 'loaded' -or $diagnostic.policy_load_s -ge 120) { throw 'Model loading not healthy enough to resume' }
    $previousPlan=Get-Content -LiteralPath (Join-Path $batch 'pilot-finalization-r1-plan.json') -Raw | ConvertFrom-Json
    $plan=@($previousPlan.runs | ForEach-Object {
        $item=@{seed=$_.seed;mode=$_.mode;name=$_.name}
        if ($_.prior_attempt) { $item.prior_attempt=$_.prior_attempt }
        $item
    })
}
if ($ResumeFinalizationFix -or $ResumeStartupAfterIOCheck) {
    if (-not (Test-Path -LiteralPath (Join-Path $batch 'pilot-plan.json'))) { throw 'Original plan missing' }
    $selected=@()
    $pendingPlan=@()
    foreach ($run in $plan) {
        $priorSummary=Join-Path $batch ($run.name+'/summary.json')
        if (Test-Path -LiteralPath $priorSummary) {
            $prior=Get-Content -LiteralPath $priorSummary -Raw | ConvertFrom-Json
            if ($prior.execution_valid) { $selected+=$run; continue }
            $knownFailure=if ($ResumeStartupAfterIOCheck) {
                $prior.error -eq 'startup_timeout' -and -not $prior.steps
            } else {
                $prior.worker_exit_code -eq -15 -and $prior.stop_reason -eq 'time_limit' -and $prior.release -eq 'neutral_command_sent'
            }
            if (-not $knownFailure -or $prior.process_cleanup.remaining_live_pids.Count -ne 0) {
                throw ('Unexpected failure requires investigation: '+$run.name)
            }
            $run=@{seed=$run.seed;mode=$run.mode;name=($run.name+$tag);prior_attempt=$run.name}
        }
        $selected+=$run
        $pendingPlan+=$run
    }
    $plan=$selected
}
$manifestPath=Join-Path $batch ('pilot'+$tag+'-plan.json')
if (Test-Path -LiteralPath $manifestPath) { throw 'Pilot already started; inspect existing evidence instead of rerunning.' }
@{schema=1;protocol=$(if ($ResumeStartupAfterIOCheck) { 'P1.2' } elseif ($ResumeFinalizationFix) { 'P1.1' } else { 'P1' });seeds=$seedList;runs=$plan;execution_runs=$pendingPlan;limit_s=180;split='development';optimizer_updates=0;scope='P1 frozen goal conditioning; declared technical retries only'} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$snapshot=Join-Path $batch ('source-at-game'+$tag)
New-Item -ItemType Directory -Path $snapshot -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'src') -Destination $snapshot -Recurse
Copy-Item -LiteralPath (Join-Path $root 'configs/experiment.json') -Destination $snapshot
Copy-Item -LiteralPath $PSCommandPath -Destination $snapshot
$sourceHashes=Get-ChildItem -LiteralPath (Join-Path $snapshot 'src/minecraftrl') -File |
    ForEach-Object { @{path=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} }
@{git_head=(git -C $root rev-parse HEAD);source=$sourceHashes;protocol_sha256=(Get-FileHash -LiteralPath (Join-Path $root 'docs/specs/2026-10-31-stage1-rm-conditioned-rl-spec.md') -Algorithm SHA256).Hash} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch ('work'+$tag+'.json')) -Encoding utf8
foreach ($run in $pendingPlan) {
    if (Test-Path -LiteralPath $stop) { Write-Output 'STOP observed; remaining conditions are unexecuted.'; break }
    $output="$linuxRoot/results/rm-conditioning-20261031/$($run.name)"
    $argsForWsl=@('-d',$Distribution,'--','env',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',
        "PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache",
        "$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python','-m','minecraftrl.cli',
        'episode','--root',$linuxRoot,'--policy','steve1','--condition','wood',
        '--seed',([string]$run.seed),'--limit-s','180','--rm-conditioning',$run.mode,
        '--rm-goal-bank',"$linuxRoot/results/rm-conditioning-20261031/goal-bank",'--output',$output,
        '--stop-file',"$linuxRoot/results/STOP")
    $argsForWsl | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($run.name+'-command.json')) -Encoding utf8
    Write-Output ('Starting '+$run.name)
    & wsl.exe @argsForWsl *> (Join-Path $batch ($run.name+'-console.log'))
    $runExit=$LASTEXITCODE
    $summaryPath=Join-Path $batch ($run.name+'/summary.json')
    if (-not (Test-Path -LiteralPath $summaryPath)) { throw ('No supervisor summary: '+$run.name+' exit '+$runExit) }
    $summary=Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    Write-Output ($run.name+': '+$summary.status+' steps='+$summary.steps+' server_goal='+$summary.server_target_reached)
    if ($runExit -ne 0 -or -not $summary.execution_valid) {
        Write-Output 'Technical error/cancellation; preserve evidence and leave remaining runs unexecuted.'
        break
    }
}
Write-Output 'Pilot driver ended. Audit summaries before interpreting performance.'
