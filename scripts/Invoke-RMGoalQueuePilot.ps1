[CmdletBinding()]
param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$batch=Join-Path $root 'results/rm-goal-queue-20261031'
$stop=Join-Path $root 'results/STOP'
$linuxRoot='/mnt/d/DATA/Desktop/MinecraftRL'
if (Test-Path -LiteralPath (Join-Path $batch 'work.json')) { throw 'P4 already prepared; no automatic rerun' }
$tests=Get-Content -LiteralPath (Join-Path $batch 'tests-wsl.log') -Raw
if ($tests -notmatch '(?m)^OK\s*$' -or $tests -match '(?m)^FAILED') { throw 'WSL tests not passed' }
$bank=Join-Path $batch 'goal-bank'
New-Item -ItemType Directory -Path $bank | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'results/rm-logo-conditioning-20261031/goal-bank') -Destination (Join-Path $bank 'wood') -Recurse
Copy-Item -LiteralPath (Join-Path $root 'weights/steve1/visual-prompts/dirt.npy') -Destination $bank
$programDir=Join-Path $batch 'programs'
New-Item -ItemType Directory -Path $programDir | Out-Null
function Write-Program([string]$Name,[string[]]$Kinds,[int]$Stall,[int]$Recovery) {
    $goals=@();$previous=$null;$index=0
    foreach ($kind in $Kinds) {
        $items=switch ($kind) { wood { @('oak_log','birch_log','spruce_log','jungle_log','acacia_log','dark_oak_log') } dirt { @('dirt') } stone { @('cobblestone') } }
        $skill=switch ($kind) { wood { 'wood_memory' } dirt { 'dirt_fixed' } stone { 'stone_collect' } }
        $requires=@()
        if ($kind -eq 'stone') { $requires=@(@{any_of=@('wooden_pickaxe','stone_pickaxe','iron_pickaxe','diamond_pickaxe','netherite_pickaxe');count=1}) }
        $id=$kind+'-'+$index
        $dependencies=if ($null -eq $previous) { @() } else { @($previous) }
        $goals+=@{id=$id;objective=('Acquire one new '+$kind+' unit as the next goal in this program.');priority=(80-$index);depends_on=@($dependencies);items=@($items);count=1;skill=$skill;requires=@($requires)}
        $previous=$id;$index++
    }
    $relative='results/rm-goal-queue-20261031/programs/'+$Name+'.json'
    @{schema='goal_queue_v1';goals=$goals;progress_limits=@{stall_ticks=$Stall;recovery_ticks=$Recovery}} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root $relative) -Encoding utf8
    return $relative
}
$programs=@{}
$programs.diagWood=Write-Program 'diag-wood' @('wood') 1200 300
$programs.diagWoodDirt=Write-Program 'diag-wood-dirt' @('wood','dirt') 1200 300
$programs.diagWoodStone=Write-Program 'diag-wood-stone' @('wood','stone') 1200 300
$programs.diagWoodTwice=Write-Program 'diag-wood-twice' @('wood','wood') 1200 300
$programs.wood=Write-Program 's1-wood' @('wood') 3600 3600
$programs.dirt=Write-Program 's1-dirt' @('dirt') 3600 3600
$programs.woodDirt=Write-Program 's1-wood-dirt' @('wood','dirt') 3600 3600
$programs.dirtWood=Write-Program 's1-dirt-wood' @('dirt','wood') 3600 3600
$programs.woodTwice=Write-Program 's1-wood-twice' @('wood','wood') 3600 3600
$runs=@()
$diagSeeds=@(81031001,81031007,81031003,81031009,81031013)
$diagPrograms=@('diagWood','diagWood','diagWoodDirt','diagWoodStone','diagWoodTwice')
for ($i=0;$i -lt 5;$i++) { $runs+=@{name=('diag-'+$diagSeeds[$i]);seed=$diagSeeds[$i];role='diagnostic';program=$programs[$diagPrograms[$i]];record=$false;variant='official'} }
$trainSeeds=@(81032001,81032003,81032007,81032009)
for ($i=0;$i -lt 4;$i++) { $key=if ($i%2 -eq 0) {'wood'} else {'dirt'}; $runs+=@{name=('train-'+$trainSeeds[$i]);seed=$trainSeeds[$i];role='train';program=$programs[$key];record=$true;variant='initial'} }
$evalSeeds=@(81033001,81033003,81033007,81033009,81033013)
$evalPrograms=@('wood','dirt','woodDirt','dirtWood','woodTwice')
for ($i=0;$i -lt 5;$i++) {
    $order=if ($i%2 -eq 0) { @('initial','candidate') } else { @('candidate','initial') }
    foreach ($variant in $order) { $runs+=@{name=('development-'+$evalSeeds[$i]+'-'+$variant);seed=$evalSeeds[$i];role='development';program=$programs[$evalPrograms[$i]];record=$true;variant=$variant} }
}
@{protocol='P4';runs=$runs;limit_s=180;training_gate='four valid new train runs, probability reproduction, nonconstant advantages';API_calls=0;automatic_promotion=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $batch 'plan.json') -Encoding utf8
$snapshot=Join-Path $batch 'source-at-game'
New-Item -ItemType Directory -Path $snapshot | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'src') -Destination $snapshot -Recurse
Copy-Item -LiteralPath (Join-Path $root 'tests') -Destination $snapshot -Recurse
$extras=@('scripts/train_s1_heads.py','scripts/probe_rm_conditioning_stop.py','scripts/Invoke-RMGoalQueuePilot.ps1','configs/experiment.json','configs/goals-wood-then-stone.json','docs/specs/2026-10-31-persistent-rm-goal-queue.md','docs/specs/2026-10-31-queue-s1-p4.md')
foreach ($relative in $extras) { Copy-Item -LiteralPath (Join-Path $root $relative) -Destination $snapshot }
Copy-Item -LiteralPath $programDir -Destination $snapshot -Recurse
$locked=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/minecraftrl') -File | Where-Object { $_.Extension -in '.py','.java' } | ForEach-Object { $_.FullName })
$locked+=@($extras | ForEach-Object { Join-Path $root $_ })
$locked+=@(Get-ChildItem -LiteralPath $programDir -File | ForEach-Object { $_.FullName })
$locked+=@(Get-ChildItem -LiteralPath $bank -Recurse -File | ForEach-Object { $_.FullName })
$locked+=@(Get-ChildItem -LiteralPath (Join-Path $root 'weights/steve1') -Filter '*.receipt.json' -File | ForEach-Object { $_.FullName })
$hashes=@($locked | ForEach-Object { @{path=$_.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
@{git_head=(git -C $root rev-parse HEAD);modified_worktree=$true;locked=$hashes;source_snapshot='source-at-game';research_spec_sha256=(Get-FileHash -LiteralPath (Join-Path $root 'docs/research-spec.md') -Algorithm SHA256).Hash} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
function Assert-Locked {
    if (Test-Path -LiteralPath $stop) { throw 'STOP is latched; do not start another run' }
    foreach ($entry in $hashes) { if ((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256) { throw ('Locked artifact changed: '+$entry.path) } }
}
if ((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') { throw 'Unrecognized STOP; preserve it' }
$priorHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
$prior=Join-Path $batch 'prior-STOP.txt'
Move-Item -LiteralPath $stop -Destination $prior
if ((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $priorHash) {
    if (-not (Test-Path -LiteralPath $stop)) { Copy-Item -LiteralPath $prior -Destination $stop }
    throw 'STOP changed during claim'
}
@{authorization='Current user approved improvements and goal queue';claimed_once=$true;prior_stop_sha256=$priorHash;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache","$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python')
$completed=@();$trained=$false;$candidateSha=$null
try {
    foreach ($run in $runs) {
        Assert-Locked
        if ($run.role -eq 'development' -and -not $trained) {
            $trainRuns=@($runs | Where-Object {$_.role -eq 'train'} | ForEach-Object { "$linuxRoot/results/rm-goal-queue-20261031/"+$_.name })
            $argv=$prefix+@("$linuxRoot/scripts/train_s1_heads.py",'--root',$linuxRoot,'--runs')+$trainRuns+@('--output',"$linuxRoot/results/rm-goal-queue-20261031/training",'--stop-file',"$linuxRoot/results/STOP")
            $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'training-command.json') -Encoding utf8
            Write-Output 'P4: starting bounded offline head PPO'
            & wsl.exe @argv *> (Join-Path $batch 'training-console.log')
            if ($LASTEXITCODE -ne 0) { throw 'Training gate/update failed; preserve all data and leave evaluation unexecuted' }
            $training=Get-Content -LiteralPath (Join-Path $batch 'training/training.json') -Raw | ConvertFrom-Json
            $candidateSha=(Get-FileHash -LiteralPath (Join-Path $batch 'training/checkpoint.pt') -Algorithm SHA256).Hash
            if (-not $training.actor_changed -or $training.checkpoint_sha256 -ne $candidateSha) { throw 'Unverified training output' }
            $trained=$true
        }
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linuxRoot,'--policy','steve1','--condition','wood','--seed',[string]$run.seed,'--limit-s','180','--rm-conditioning','goal-queue','--rm-goal-bank',"$linuxRoot/results/rm-goal-queue-20261031/goal-bank",'--rm-program',"$linuxRoot/$($run.program)",'--output',"$linuxRoot/results/rm-goal-queue-20261031/$($run.name)",'--stop-file',"$linuxRoot/results/STOP")
        if ($run.record) { $argv+=@('--s1-record','--s1-split',$run.role) }
        if ($run.variant -eq 'candidate') {
            if ((Get-FileHash -LiteralPath (Join-Path $batch 'training/checkpoint.pt') -Algorithm SHA256).Hash -ne $candidateSha) { throw 'Candidate weights changed' }
            $argv+=@('--s1-checkpoint',"$linuxRoot/results/rm-goal-queue-20261031/training/checkpoint.pt")
        }
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($run.name+'-command.json')) -Encoding utf8
        Write-Output ('P4 starting '+$run.name)
        & wsl.exe @argv *> (Join-Path $batch ($run.name+'-console.log'))
        $exitCode=$LASTEXITCODE
        $summary=Get-Content -LiteralPath (Join-Path $batch ($run.name+'/summary.json')) -Raw | ConvertFrom-Json
        $errors=@($summary.PSObject.Properties.Name | Where-Object { $_ -in 'error','environment_close_error','policy_close_error','finalization_marker_error' })
        $pause=($run.role -eq 'diagnostic' -and $summary.status -eq 'planning_paused' -and $summary.worker_exit_code -eq 2 -and $summary.release -eq 'neutral_command_sent')
        if ($errors.Count -or $summary.process_cleanup.remaining_live_pids.Count -or (-not $pause -and (-not $summary.execution_valid -or $summary.worker_exit_code -ne 0 -or $exitCode -ne 0))) { throw ('Technical run failure: '+$run.name) }
        $completed+=@{name=$run.name;status=$summary.status;completed_goals=$summary.completed_goal_count;steps=$summary.steps;worker_exit_code=$summary.worker_exit_code}
        $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output ('P4 finished '+$run.name+': '+$summary.status+'; goals='+$summary.completed_goal_count+'; steps='+$summary.steps)
    }
    Assert-Locked
    $argv=$prefix+@("$linuxRoot/scripts/probe_rm_conditioning_stop.py",'--root',$linuxRoot,'--batch','rm-goal-queue-20261031','--mode','goal-queue','--program',"$linuxRoot/$($programs.wood)",'--s1-record')
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-helper-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if ($LASTEXITCODE -ne 0) { throw 'STOP diagnostic failed; preserve evidence' }
    @{completed_runs=$completed.Count;training_executed=$trained;candidate_sha256=$candidateSha;automatic_promotion=$false;stop_probe_passed=$true;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} finally {
    if (-not (Test-Path -LiteralPath $stop)) { 'P4 goal queue batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8 }
}
