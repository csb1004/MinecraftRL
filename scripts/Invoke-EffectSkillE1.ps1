param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL'; $linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='effect-skill-e1-20261102'; $batch=Join-Path $root "results/$name"; $stop=Join-Path $root 'results/STOP'
if(Test-Path -LiteralPath (Join-Path $batch 'work.json')){throw 'E1 already claimed; do not rerun'}
$plan=Get-Content -LiteralPath (Join-Path $batch 'plan.json') -Raw | ConvertFrom-Json
if($plan.runs.Count -ne 14){throw 'Expected 14 fixed cases'}
$tests=Get-Content -LiteralPath (Join-Path $batch 'tests-wsl-final.log') -Raw
if($tests -notmatch '(?m)^OK\s*$' -or $tests -match '(?m)^FAILED'){throw 'Relevant tests have not passed'}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') {throw 'Unrecognized STOP; preserve it'}
$sources=@(foreach($folder in @('src','scripts','tests')){
    Get-ChildItem -LiteralPath (Join-Path $root $folder) -Recurse -File | Where-Object {$_.Extension -in '.py','.java','.ps1','.sh'}
})
$sources+=Get-Item -LiteralPath (Join-Path $root 'configs/experiment.json'),(Join-Path $root 'configs/logo-wood-validators-v1.json'),(Join-Path $root 'docs/specs/2026-11-02-effect-skill-e1.md')
$locked=@(foreach($file in $sources | Sort-Object FullName -Unique){
    $relative=$file.FullName.Substring($root.Length+1).Replace('\','/')
    $destination=Join-Path $batch "source-at-game/$relative"
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    @{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
})
$inputFiles=@(foreach($folder in @('goal-bank','programs')){Get-ChildItem -LiteralPath (Join-Path $batch $folder) -Recurse -File})
$inputFiles+=Get-Item -LiteralPath (Join-Path $batch 'plan.json')
foreach($world in $plan.runs.world | Sort-Object -Unique){
    $inputFiles+=Get-Item -LiteralPath (Join-Path $root $world),(Join-Path $root ([IO.Path]::ChangeExtension($world,'json')))
}
$inputs=@(foreach($file in $inputFiles){@{path=$file.FullName.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}})
$official=@(foreach($path in @('weights/steve1/steve1.weights','weights/steve1/steve1_prior.pt','weights/steve1/attn.pth','weights/steve1/2x.model','docs/research-spec.md')){
    @{path=$path;sha256=(Get-FileHash -LiteralPath (Join-Path $root $path) -Algorithm SHA256).Hash}
})
@{locked=$locked;inputs=$inputs;official=$official;git_head=(git -C $root rev-parse HEAD);source_snapshot='source-at-game';modified_worktree=$true} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
function Assert-Locked {
    if(Test-Path -LiteralPath $stop){throw 'STOP latched; no next run'}
    foreach($entry in ($locked+$inputs)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Frozen input changed: $($entry.path)"}
    }
}
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linux/src","XDG_CACHE_HOME=$linux/.cache","$linux/.runtime/bin/micromamba",'run','-p',"$linux/.runtime/steve1",'python')
$completed=@(); $trained=$false; $candidateHash=$null
$env:PYTHONPATH=Join-Path $root 'src'
try {
    $prior=Join-Path $batch 'prior-STOP.txt'
    $stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
    Move-Item -LiteralPath $stop -Destination $prior
    if((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $stopHash){throw 'STOP changed during claim'}
    @{utc=[DateTime]::UtcNow.ToString('o');prior_stop_sha256=$stopHash;authorization='User authorized general RM-to-RL structural improvement and local validation';claimed_once=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
    foreach($r in $plan.runs){
        Assert-Locked
        if($r.role -eq 'development' -and -not $trained){
            & "$root/.runtime/fixture-tools/Scripts/python.exe" "$root/scripts/analyze_effect_skills.py" $batch *> (Join-Path $batch 'train-raw-audit.log')
            if($LASTEXITCODE -ne 0){throw 'Train raw audit failed'}
            $argv=$prefix+@("$linux/scripts/train_effect_adapter.py",'--root',$linux,'--output',"$linux/results/$name/training",'--stop-file',"$linux/results/STOP",'--variant','nonlinear-on','--runs')+@($plan.runs | Where-Object {$_.role -eq 'train'} | ForEach-Object {"$linux/results/$name/$($_.name)"})
            $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'training-command.json') -Encoding utf8
            Write-Output 'E1 training from newly verified train trajectories'
            & wsl.exe @argv *> (Join-Path $batch 'training-console.log')
            if($LASTEXITCODE -ne 0){throw 'E1 training failed'}
            $candidateHash=(Get-FileHash -LiteralPath (Join-Path $batch 'training/checkpoint.pt') -Algorithm SHA256).Hash
            $trained=$true
        }
        if($r.candidate -and (Get-FileHash -LiteralPath (Join-Path $root $r.checkpoint) -Algorithm SHA256).Hash -ne $candidateHash){throw 'Candidate identity changed'}
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$r.seed,'--limit-s',[string]$r.limit_s,'--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/$name/programs/$($r.program).json",'--output',"$linux/results/$name/$($r.name)",'--stop-file',"$linux/results/STOP",'--effect-learning','--recovery-variant',$r.variant,'--recovery-world',"$linux/$($r.world)",'--recovery-task',$r.task,'--recovery-split',$r.role,'--recovery-action-budget',[string]$r.action_budget,'--recovery-sampling-seed',[string]$r.sampling_seed)
        if($r.candidate){$argv+=@('--recovery-checkpoint',"$linux/$($r.checkpoint)")}
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($r.name+'-command.json')) -Encoding utf8
        Write-Output "E1 starting $($r.name)"
        & wsl.exe @argv *> (Join-Path $batch ($r.name+'-console.log'))
        if($LASTEXITCODE -ne 0){throw "Execution failed: $($r.name)"}
        $s=Get-Content -LiteralPath (Join-Path $batch "$($r.name)/summary.json") -Raw | ConvertFrom-Json
        if(-not $s.execution_valid -or $s.worker_exit_code -ne 0 -or $s.process_cleanup.remaining_live_pids.Count -or $s.environment_close_error -or $s.policy_close_error){throw "Invalid execution: $($r.name)"}
        $completed+=@{name=$r.name;steps=$s.steps;completed=$s.program_completed;effects=$s.effect_successes;status=$s.status}
        $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output "E1 finished $($r.name): completed=$($s.program_completed), effects=$($s.effect_successes), steps=$($s.steps)"
    }
    Assert-Locked
    & "$root/.runtime/fixture-tools/Scripts/python.exe" "$root/scripts/analyze_effect_skills.py" $batch *> (Join-Path $batch 'final-raw-audit.log')
    if($LASTEXITCODE -ne 0){throw 'Final raw audit failed'}
    $argv=$prefix+@("$linux/scripts/probe_rm_conditioning_stop.py",'--root',$linux,'--batch',$name,'--mode','goal-queue','--program',"$linux/results/$name/programs/stop.json",'--effect-learning','--recovery-variant','nonlinear-on','--recovery-checkpoint',"$linux/results/$name/training/checkpoint.pt",'--recovery-world',"$linux/results/common-recovery-r1-20261101/worlds/natural-81112001.zip")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if($LASTEXITCODE -ne 0){throw 'Actual E1 STOP diagnostic failed'}
    foreach($entry in ($locked+$inputs+$official)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Final identity changed: $($entry.path)"}
    }
    @{utc=[DateTime]::UtcNow.ToString('o');completed_runs=$completed.Count;training_jobs=1;candidate_sha256=$candidateHash;stop_passed=$true;official_unchanged=$true;automatic_promotion=$false;API_calls=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} finally {
    if(-not(Test-Path -LiteralPath $stop)){'E1 batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
