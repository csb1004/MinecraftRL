[CmdletBinding()]
param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$name='rm-general-s2-20261101'
$batch=Join-Path $root "results/$name"
$stop=Join-Path $root 'results/STOP'
$linuxRoot='/mnt/d/DATA/Desktop/MinecraftRL'
if ($root -ne 'D:\DATA\Desktop\MinecraftRL') { throw 'Actual project path required' }
if (Test-Path -LiteralPath (Join-Path $batch 'work.json')) { throw 'S2 already prepared; inspect evidence before a separate amendment' }
$tests=Get-Content -LiteralPath (Join-Path $batch 'tests-wsl.log') -Raw
if ($tests -notmatch '(?m)^OK\s*$' -or $tests -match '(?m)^FAILED') { throw 'S2 tests not passed' }
$bank=Join-Path $batch 'goal-bank'
Copy-Item -LiteralPath (Join-Path $root 'results/rm-goal-queue-20261031/goal-bank') -Destination $bank -Recurse
$programDir=Join-Path $batch 'programs'
New-Item -ItemType Directory -Path $programDir | Out-Null
function Write-Program([string]$Id,[string[]]$Kinds,[int[]]$Counts) {
    $goals=@();$previous=$null
    for ($i=0;$i -lt $Kinds.Count;$i++) {
        $kind=$Kinds[$i];$key=$kind+'-'+$i
        $items=if ($kind -eq 'wood') { @('oak_log','birch_log','spruce_log','jungle_log','acacia_log','dark_oak_log') } else { @('dirt') }
        $skill=if ($kind -eq 'wood') {'wood_memory'} else {'dirt_fixed'}
        $deps=if ($null -eq $previous) {@()} else {@($previous)}
        $goals+=@{id=$key;objective="Acquire $($Counts[$i]) new $kind units.";priority=(80-$i);depends_on=@($deps);items=@($items);count=$Counts[$i];skill=$skill;requires=@()}
        $previous=$key
    }
    $relative="results/$name/programs/$Id.json"
    @{schema='goal_queue_v1';goals=$goals;progress_limits=@{stall_ticks=600;recovery_ticks=300}} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root $relative) -Encoding utf8
    return $relative
}
$programs=@{}
foreach ($kind in @('wood','dirt')) { foreach ($count in @(1,2,3)) { $programs["$kind$count"]=Write-Program "$kind$count" @($kind) @($count) } }
$programs['woodDirt']=Write-Program 'wood-dirt' @('wood','dirt') @(1,1)
$programs['dirtWood']=Write-Program 'dirt-wood' @('dirt','wood') @(1,1)
$trainSeeds=@(81101001,81101003,81101007,81101009,81101013,81101017,81101019,81101023,81101029,81101031,81101037,81101041)
$trainPrograms=@('wood1','dirt1','wood2','wood1','dirt2','wood2','wood1','dirt1','wood2','wood1','dirt2','wood2')
$runs=@()
for ($i=0;$i -lt $trainSeeds.Count;$i++) { $runs+=@{name="train-$($trainSeeds[$i])";seed=$trainSeeds[$i];role='train';program=$programs[$trainPrograms[$i]];variant='linear-on';candidate=$false} }
$evalSeeds=@(81102001,81102003,81102007,81102009,81102013)
$evalPrograms=@('wood1','dirt1','wood3','woodDirt','dirtWood')
$variants=@('initial','linear-off','linear-on','nonlinear-off','nonlinear-on')
for ($i=0;$i -lt $evalSeeds.Count;$i++) {
    for ($j=0;$j -lt $variants.Count;$j++) {
        $v=$variants[($i+$j)%$variants.Count]
        $runs+=@{name="development-$($evalSeeds[$i])-$v";seed=$evalSeeds[$i];role='development';program=$programs[$evalPrograms[$i]];variant=$(if ($v -eq 'initial') {'linear-on'} else {$v});candidate=($v -ne 'initial')}
    }
}
@{protocol='S2';runs=$runs;limit_s=180;action_budget=1200;API_calls=0;automatic_promotion=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $batch 'plan.json') -Encoding utf8
$snapshot=Join-Path $batch 'source-at-game'
New-Item -ItemType Directory -Path $snapshot | Out-Null
$extras=@('scripts/train_s2_conditions.py','scripts/probe_rm_conditioning_stop.py','scripts/Invoke-RMGeneralS2.ps1','docs/specs/2026-11-01-general-rm-s2.md','configs/experiment.json')
Copy-Item -LiteralPath (Join-Path $root 'src') -Destination $snapshot -Recurse
Copy-Item -LiteralPath (Join-Path $root 'tests') -Destination $snapshot -Recurse
foreach ($relative in $extras) { Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $snapshot }
Copy-Item -LiteralPath $programDir -Destination $snapshot -Recurse
$locked=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/minecraftrl') -File | Where-Object {$_.Extension -in '.py','.java'} | ForEach-Object {$_.FullName})
$locked+=@($extras | ForEach-Object {Join-Path $root $_})
$locked+=@(Get-ChildItem -LiteralPath $programDir -File | ForEach-Object {$_.FullName})
$locked+=@(Get-ChildItem -LiteralPath $bank -File -Recurse | ForEach-Object {$_.FullName})
$hashes=@($locked | ForEach-Object {@{path=$_.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash}})
$official=@('weights/steve1/steve1.weights','weights/steve1/steve1_prior.pt','weights/steve1/attn.pth','weights/steve1/2x.model','docs/research-spec.md')
$officialHashes=@($official | ForEach-Object {@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $root $_) -Algorithm SHA256).Hash}})
@{git_head=(git -C $root rev-parse HEAD);modified_worktree=$true;locked=$hashes;official=$officialHashes;source_snapshot='source-at-game'} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
function Assert-Locked {
    if (Test-Path -LiteralPath $stop) { throw 'STOP latched; no further run' }
    foreach ($entry in $hashes) { if ((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Locked source changed: $($entry.path)" } }
}
if ((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') { throw 'Unrecognized STOP; preserve it' }
$stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
$prior=Join-Path $batch 'prior-STOP.txt'
Move-Item -LiteralPath $stop -Destination $prior
if ((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $stopHash) {
    if (-not (Test-Path -LiteralPath $stop)) {Copy-Item -LiteralPath $prior -Destination $stop}
    throw 'STOP changed during claim'
}
@{authorization='User explicitly approved generalized improvements and real-game validation on 2026-11-01';prior_stop_sha256=$stopHash;claimed_once=$true;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache","$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python')
$completed=@();$trained=$false;$candidateHashes=@{}
try {
    foreach ($run in $runs) {
        Assert-Locked
        if ($run.role -eq 'development' -and -not $trained) {
            $trainPaths=@($runs | Where-Object {$_.role -eq 'train'} | ForEach-Object {"$linuxRoot/results/$name/$($_.name)"})
            $argv=$prefix+@("$linuxRoot/scripts/train_s2_conditions.py",'--root',$linuxRoot,'--runs')+$trainPaths+@('--output',"$linuxRoot/results/$name/training",'--stop-file',"$linuxRoot/results/STOP")
            $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'training-command.json') -Encoding utf8
            Write-Output 'S2: verify common behavior probabilities and train four candidates'
            & wsl.exe @argv *> (Join-Path $batch 'training-console.log')
            if ($LASTEXITCODE -ne 0) {throw 'S2 training failed; preserve evidence'}
            foreach ($v in @('linear-off','linear-on','nonlinear-off','nonlinear-on')) { $candidateHashes[$v]=(Get-FileHash -LiteralPath (Join-Path $batch "training/$v/checkpoint.pt") -Algorithm SHA256).Hash }
            $trained=$true
        }
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linuxRoot,'--policy','steve1','--condition','wood','--seed',[string]$run.seed,'--limit-s','180','--rm-conditioning','goal-queue','--rm-goal-bank',"$linuxRoot/results/$name/goal-bank",'--rm-program',"$linuxRoot/$($run.program)",'--output',"$linuxRoot/results/$name/$($run.name)",'--stop-file',"$linuxRoot/results/STOP",'--s2-variant',$run.variant,'--s2-split',$run.role,'--s2-action-budget','1200')
        if ($run.candidate) {
            if ((Get-FileHash -LiteralPath (Join-Path $batch "training/$($run.variant)/checkpoint.pt") -Algorithm SHA256).Hash -ne $candidateHashes[$run.variant]) {throw 'Candidate changed'}
            $argv+=@('--s2-checkpoint',"$linuxRoot/results/$name/training/$($run.variant)/checkpoint.pt")
        }
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($run.name+'-command.json')) -Encoding utf8
        Write-Output "S2 starting $($run.name)"
        & wsl.exe @argv *> (Join-Path $batch ($run.name+'-console.log'))
        $code=$LASTEXITCODE
        $summary=Get-Content -LiteralPath (Join-Path $batch "$($run.name)/summary.json") -Raw | ConvertFrom-Json
        $errors=@($summary.PSObject.Properties.Name | Where-Object {$_ -in 'error','environment_close_error','policy_close_error','finalization_marker_error'})
        if ($code -ne 0 -or -not $summary.execution_valid -or $summary.worker_exit_code -ne 0 -or $errors.Count -or $summary.process_cleanup.remaining_live_pids.Count) {throw "Technical failure: $($run.name)"}
        $completed+=@{name=$run.name;status=$summary.status;goals=$summary.completed_goal_count;steps=$summary.steps;reason=$summary.s2_end_reason;worker_exit_code=$summary.worker_exit_code}
        $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output "S2 finished $($run.name): $($summary.status); goals=$($summary.completed_goal_count); steps=$($summary.steps); reason=$($summary.s2_end_reason)"
    }
    Assert-Locked
    $argv=$prefix+@("$linuxRoot/scripts/probe_rm_conditioning_stop.py",'--root',$linuxRoot,'--batch',$name,'--mode','goal-queue','--program',"$linuxRoot/$($programs['wood1'])",'--s2-variant','nonlinear-on','--s2-checkpoint',"$linuxRoot/results/$name/training/nonlinear-on/checkpoint.pt")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-helper-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if ($LASTEXITCODE -ne 0) {throw 'Actual S2 STOP diagnostic failed'}
    foreach ($entry in $officialHashes) {if ((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256) {throw "Official artifact changed: $($entry.path)"}}
    @{completed_runs=$completed.Count;training_executed=$trained;candidates=$candidateHashes;stop_probe_passed=$true;automatic_promotion=$false;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} finally {
    if (-not (Test-Path -LiteralPath $stop)) {'S2 generalized condition batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
