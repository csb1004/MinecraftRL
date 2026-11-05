param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL'; $linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='execution-contract-e2-20261103'; $batch=Join-Path $root "results/$name"; $stop=Join-Path $root 'results/STOP'
if(Test-Path -LiteralPath (Join-Path $batch 'work.json')){throw 'E2 already claimed; do not rerun'}
$preparation=Get-Content -LiteralPath (Join-Path $batch 'preparation-plan.json') -Raw | ConvertFrom-Json
$tests=Get-Content -LiteralPath (Join-Path $batch 'tests-wsl-final.log') -Raw
if($tests -notmatch '(?m)^OK\s*$' -or $tests -match '(?m)^FAILED'){throw 'Relevant tests have not passed'}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'E1 supplemental check ended; do not resume automatically.') {throw 'Unrecognized STOP; preserve it'}
$sources=@(foreach($folder in @('src','scripts','tests')){
    Get-ChildItem -LiteralPath (Join-Path $root $folder) -Recurse -File | Where-Object {$_.Extension -in '.py','.java','.ps1','.sh'}
})
$sources+=Get-Item -LiteralPath (Join-Path $root 'configs/experiment.json'),(Join-Path $root 'configs/logo-wood-validators-v1.json'),(Join-Path $root 'docs/specs/2026-11-03-execution-contract-e2.md')
$locked=@(foreach($file in $sources | Sort-Object FullName -Unique){
    $relative=$file.FullName.Substring($root.Length+1).Replace('\','/')
    $destination=Join-Path $batch "source-at-game/$relative"
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    @{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
})
$inputFiles=@(foreach($folder in @('goal-bank','programs')){Get-ChildItem -LiteralPath (Join-Path $batch $folder) -Recurse -File})
$inputFiles+=Get-Item -LiteralPath (Join-Path $batch 'preparation-plan.json'),(Join-Path $batch 'tests-wsl-final.log')
$inputs=@(foreach($file in $inputFiles){@{path=$file.FullName.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}})
$official=@(foreach($path in @('weights/steve1/steve1.weights','weights/steve1/steve1_prior.pt','weights/steve1/attn.pth','weights/steve1/2x.model','docs/research-spec.md','results/effect-skill-e1-20261102/driver-completion.json','results/effect-skill-e1-20261103-resource-check/driver-completion.json')){
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
$completed=@();$trained=$false;$candidateHashes=@{};$inputsFinal=$false
$env:PYTHONPATH=Join-Path $root 'src'
try {
    $prior=Join-Path $batch 'prior-STOP.txt';$stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
    Move-Item -LiteralPath $stop -Destination $prior
    if((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $stopHash){throw 'STOP changed during claim'}
    @{utc=[DateTime]::UtcNow.ToString('o');prior_stop_sha256=$stopHash;authorization='User approved four concrete RM-to-RL improvements and necessary local validation';claimed_once=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
    foreach($seed in $preparation.preparation_seeds){
        Assert-Locked
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$seed,'--limit-s','180','--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/common-recovery-r1-20261101/programs/wood1.json",'--recovery-prepare','--output',"$linux/results/$name/prepare-$seed",'--stop-file',"$linux/results/STOP")
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch "prepare-$seed-command.json") -Encoding utf8
        Write-Output "E2 preparing neutral train start $seed"
        & wsl.exe @argv *> (Join-Path $batch "prepare-$seed-console.log")
        if($LASTEXITCODE -ne 0){throw "Preparation failed: $seed"}
        $s=Get-Content -LiteralPath (Join-Path $batch "prepare-$seed/summary.json") -Raw | ConvertFrom-Json
        if($s.status -ne 'world_prepared' -or -not $s.execution_valid -or $s.learned_policy_loaded -or $s.process_cleanup.remaining_live_pids.Count){throw 'Preparation invalid'}
    }
    & "$root/.runtime/fixture-tools/Scripts/python.exe" "$root/scripts/prepare_execution_pilot.py" finalize --root $root *> (Join-Path $batch 'finalize-plan.log')
    if($LASTEXITCODE -ne 0){throw 'Plan finalization failed'}
    $plan=Get-Content -LiteralPath (Join-Path $batch 'plan.json') -Raw | ConvertFrom-Json
    if($plan.runs.Count -ne 26){throw 'Expected 26 fixed cases'}
    $newInputs=@(Get-Item -LiteralPath (Join-Path $batch 'plan.json'))
    foreach($world in $plan.runs.world | Sort-Object -Unique){$newInputs+=Get-Item -LiteralPath (Join-Path $root $world),(Join-Path $root ([IO.Path]::ChangeExtension($world,'json')))}
    $inputs+=@(foreach($file in $newInputs){@{path=$file.FullName.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}})
    @{inputs=$inputs;utc=[DateTime]::UtcNow.ToString('o');before_first_policy_run=$true} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'frozen-inputs.json') -Encoding utf8
    $inputsFinal=$true
    foreach($r in $plan.runs){
        Assert-Locked
        if($r.role -eq 'development' -and -not $trained){
            & "$root/.runtime/fixture-tools/Scripts/python.exe" "$root/scripts/analyze_execution_contracts.py" $batch *> (Join-Path $batch 'train-raw-audit.log')
            if($LASTEXITCODE -ne 0){throw 'Train raw audit failed'}
            foreach($model in @('plain','calibrated','retained')){
                Assert-Locked
                $argv=$prefix+@("$linux/scripts/train_execution_adapter.py",'--root',$linux,'--output',"$linux/results/$name/training-$model",'--stop-file',"$linux/results/STOP",'--variant','nonlinear-on','--runs')+@($plan.runs | Where-Object {$_.data_role -eq 'PPO'} | ForEach-Object {"$linux/results/$name/$($_.name)"})
                if($model -ne 'plain'){$argv+=@('--critic-warmup-epochs','10')}
                if($model -eq 'retained'){$argv+=@('--retention-runs')+@($plan.runs | Where-Object {$_.data_role -eq 'official_KL_only'} | ForEach-Object {"$linux/results/$name/$($_.name)"})}
                $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch "training-$model-command.json") -Encoding utf8
                Write-Output "E2 training $model from the same fresh train trajectories"
                & wsl.exe @argv *> (Join-Path $batch "training-$model-console.log")
                if($LASTEXITCODE -ne 0){throw "E2 training failed: $model"}
                $candidateHashes[$model]=(Get-FileHash -LiteralPath (Join-Path $batch "training-$model/checkpoint.pt") -Algorithm SHA256).Hash
            }
            $trained=$true
        }
        if($r.candidate -and (Get-FileHash -LiteralPath (Join-Path $root $r.checkpoint) -Algorithm SHA256).Hash -ne $candidateHashes[$r.model]){throw 'Candidate identity changed'}
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$r.seed,'--limit-s',[string]$r.limit_s,'--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/$name/programs/$($r.program).json",'--output',"$linux/results/$name/$($r.name)",'--stop-file',"$linux/results/STOP",'--execution-learning','--recovery-variant',$r.variant,'--recovery-world',"$linux/$($r.world)",'--recovery-task',$r.task,'--recovery-split',$r.role,'--recovery-action-budget',[string]$r.action_budget,'--recovery-sampling-seed',[string]$r.sampling_seed)
        if($r.candidate){$argv+=@('--recovery-checkpoint',"$linux/$($r.checkpoint)")}
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($r.name+'-command.json')) -Encoding utf8
        Write-Output "E2 starting $($r.name)"
        & wsl.exe @argv *> (Join-Path $batch ($r.name+'-console.log'))
        if($LASTEXITCODE -ne 0){throw "Execution failed: $($r.name)"}
        $s=Get-Content -LiteralPath (Join-Path $batch "$($r.name)/summary.json") -Raw | ConvertFrom-Json
        if(-not $s.execution_valid -or $s.worker_exit_code -ne 0 -or $s.process_cleanup.remaining_live_pids.Count -or $s.environment_close_error -or $s.policy_close_error){throw "Invalid execution: $($r.name)"}
        $completed+=@{name=$r.name;steps=$s.steps;completed=$s.program_completed;status=$s.status;inventory=$s.inventory_final_delta}
        $completed | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output "E2 finished $($r.name): completed=$($s.program_completed), steps=$($s.steps)"
    }
    Assert-Locked
    & "$root/.runtime/fixture-tools/Scripts/python.exe" "$root/scripts/analyze_execution_contracts.py" $batch *> (Join-Path $batch 'final-raw-audit.log')
    if($LASTEXITCODE -ne 0){throw 'Final raw audit failed'}
    $argv=$prefix+@("$linux/scripts/probe_rm_conditioning_stop.py",'--root',$linux,'--batch',$name,'--mode','goal-queue','--program',"$linux/results/$name/programs/stop.json",'--execution-learning','--recovery-variant','nonlinear-on','--recovery-checkpoint',"$linux/results/$name/training-retained/checkpoint.pt",'--recovery-world',"$linux/results/common-recovery-r1-20261101/worlds/natural-81112001.zip")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if($LASTEXITCODE -ne 0){throw 'Actual E2 STOP diagnostic failed'}
    foreach($entry in ($locked+$inputs+$official)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Final identity changed: $($entry.path)"}
    }
    @{utc=[DateTime]::UtcNow.ToString('o');completed_runs=$completed.Count;training_jobs=3;candidate_sha256=$candidateHashes;stop_passed=$true;official_unchanged=$true;automatic_promotion=$false;API_calls=0} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} catch {
    @{utc=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message;completed_runs=$completed.Count;inputs_final=$inputsFinal;trained=$trained} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-failure.json') -Encoding utf8
    throw
} finally {
    if(-not(Test-Path -LiteralPath $stop)){'E2 batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
