param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL';$linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='common-recovery-r1-20261101';$batch=Join-Path $root "results/$name";$stop=Join-Path $root 'results/STOP'
if(Test-Path -LiteralPath (Join-Path $batch 'work.json')){throw 'Measured batch already claimed; never rerun automatically'}
if(-not(Test-Path -LiteralPath (Join-Path $batch 'preparation-completion.json'))){throw 'Neutral saved starts incomplete'}
if(-not(Test-Path -LiteralPath (Join-Path $batch 'diagnostic-completion.json'))){throw 'Actual diagnostic incomplete'}
if(-not(Test-Path -LiteralPath (Join-Path $batch 'diagnostic-review.json'))){throw 'Inspect actual restoration and frames before measured comparison'}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Common recovery diagnostic finished; do not resume automatically.') {throw 'Complete diagnostic before claiming measured batch'}
$manifest=Get-Content -LiteralPath (Join-Path $batch 'world-manifest.json') -Raw | ConvertFrom-Json
$round1=@($manifest.cases | Where-Object {$_.id -like 'train-r1-*' -or $_.id -like 'train-natural-*'})
$round2=@($manifest.cases | Where-Object {$_.id -like 'train-r2-*' -or $_.id -like 'train-natural-*'})
$development=@($manifest.cases | Where-Object {$_.split -eq 'development'})
if($round1.Count -ne 6 -or $round2.Count -ne 6 -or $development.Count -ne 10){throw 'Unexpected fixed case counts'}
$jobs=@();$runs=@()
function Game-Record($case,$role,$variant,$checkpoint,$prefix,$offset) {
    [pscustomobject]@{name="$prefix-$($case.id)";case_id=$case.id;role=$role;seed=$case.seed;
        sampling_seed=([int]$case.sampling_seed+$offset);variant=$variant;candidate=[bool]$checkpoint;
        checkpoint=$checkpoint;world=$case.world;world_sha256=$case.sha256;task=$case.task;program=$case.program;
        action_budget=$(if($case.task -eq 'pickup'){400}else{1200});limit_s=$(if($case.task -eq 'pickup'){90}else{180})}
}
foreach($case in $round1){$r=Game-Record $case 'train' 'nonlinear-on' $null 'r1-shared' 0;$runs+=$r;$jobs+=@{kind='game';run=$r}}
foreach($variant in @('nonlinear-off','nonlinear-on')){$jobs+=@{kind='train';round=1;variant=$variant;parent=$null;runs=@($runs | ForEach-Object {$_.name})}}
foreach($variant in @('nonlinear-off','nonlinear-on')) {
    $candidate="results/$name/training/round1/$variant/checkpoint.pt";$roundNames=@()
    foreach($case in $round2){$r=Game-Record $case 'train' $variant $candidate "r2-$variant" 1000;$runs+=$r;$roundNames+=$r.name;$jobs+=@{kind='game';run=$r}}
    $jobs+=@{kind='train';round=2;variant=$variant;parent=$candidate;runs=$roundNames}
}
$arms=@('initial','nonlinear-off','nonlinear-on');$index=0
foreach($case in $development) {
    foreach($shift in 0..2) {
        $arm=$arms[($index+$shift)%3];$variant=if($arm -eq 'initial'){'nonlinear-on'}else{$arm}
        $candidate=if($arm -eq 'initial'){$null}else{"results/$name/training/round2/$arm/checkpoint.pt"}
        $r=Game-Record $case 'development' $variant $candidate "dev-$arm" 0
        $runs+=$r;$jobs+=@{kind='game';run=$r}
    };$index++
}
@{protocol='common_recovery_r1';runs=$runs;jobs=$jobs;automatic_promotion=$false} | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $batch 'plan.json') -Encoding utf8
$stopProgram=Get-Content -LiteralPath (Join-Path $batch 'programs/wood1.json') -Raw | ConvertFrom-Json
$stopProgram.goals[0].count=64
$stopProgram | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $batch 'programs/stop-wood64.json') -Encoding utf8
$sourceFiles=@(Get-ChildItem -LiteralPath (Join-Path $root 'src') -Recurse -File | Where-Object {$_.Extension -in '.py','.java'})
$sourceFiles+=@(Get-ChildItem -LiteralPath (Join-Path $root 'scripts') -File | Where-Object {$_.Extension -in '.py','.ps1','.sh'})
$sourceFiles+=@(Get-ChildItem -LiteralPath (Join-Path $root 'tests') -File | Where-Object {$_.Extension -eq '.py'})
$sourceFiles+=Get-Item -LiteralPath (Join-Path $root 'configs/experiment.json'),(Join-Path $root 'configs/logo-wood-validators-v1.json'),(Join-Path $root 'docs/specs/2026-11-01-common-recovery-r1.md')
$locked=@(foreach($file in $sourceFiles | Sort-Object FullName -Unique){
    $relative=$file.FullName.Substring($root.Length+1).Replace('\','/')
    $destination=Join-Path $batch "source-at-game/$relative"
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    @{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
})
$inputs=@(foreach($folder in @('worlds','goal-bank','programs')) {
    foreach($file in Get-ChildItem -LiteralPath (Join-Path $batch $folder) -Recurse -File){
        @{path=$file.FullName.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
    }
})
$official=@(foreach($path in @('weights/steve1/steve1.weights','weights/steve1/steve1_prior.pt','weights/steve1/attn.pth','weights/steve1/2x.model','docs/research-spec.md')){
    @{path=$path;sha256=(Get-FileHash -LiteralPath (Join-Path $root $path) -Algorithm SHA256).Hash}
})
@{locked=$locked;inputs=$inputs;official=$official;git_head=(git -C $root rev-parse HEAD);source_snapshot='source-at-game';modified_worktree=$true} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
function Assert-Locked {
    if(Test-Path -LiteralPath $stop){throw 'STOP latched; no next action'}
    foreach($entry in ($locked+$inputs)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Frozen input changed: $($entry.path)"}
    }
}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Common recovery diagnostic finished; do not resume automatically.') {throw 'Complete diagnostic and inspect STOP before measured batch'}
$stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
Move-Item -LiteralPath $stop -Destination (Join-Path $batch 'pre-measurement-STOP.txt')
@{utc=[DateTime]::UtcNow.ToString('o');prior_stop_sha256=$stopHash;authorization='User-approved common recovery training and actual validation';claimed_once=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-measurement.json') -Encoding utf8
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linux/src","XDG_CACHE_HOME=$linux/.cache","$linux/.runtime/bin/micromamba",'run','-p',"$linux/.runtime/steve1",'python')
$completed=@();$candidateHashes=@{}
try {
    foreach($job in $jobs) {
        Assert-Locked
        if($job.kind -eq 'train') {
            $id="training/round$($job.round)/$($job.variant)"
            $argv=$prefix+@("$linux/scripts/train_recovery_adapter.py",'--root',$linux,'--variant',$job.variant,'--output',"$linux/results/$name/$id",'--stop-file',"$linux/results/STOP",'--runs')+@($job.runs | ForEach-Object {"$linux/results/$name/$_"})
            if($job.parent){$argv+=@('--parent',"$linux/$($job.parent)")}
            $key="train-r$($job.round)-$($job.variant)"
            Write-Output "Training $key from verified fresh behavior"
            $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch "$key-command.json") -Encoding utf8
            & wsl.exe @argv *> (Join-Path $batch "$key-console.log")
            if($LASTEXITCODE -ne 0){throw "Training failed: $key"}
            $path="results/$name/$id/checkpoint.pt";$candidateHashes[$path]=(Get-FileHash -LiteralPath (Join-Path $root $path) -Algorithm SHA256).Hash
            continue
        }
        $r=$job.run
        if($r.candidate -and (Get-FileHash -LiteralPath (Join-Path $root $r.checkpoint) -Algorithm SHA256).Hash -ne $candidateHashes[$r.checkpoint]){throw 'Candidate changed'}
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$r.seed,'--limit-s',[string]$r.limit_s,'--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/$name/programs/$($r.program).json",'--output',"$linux/results/$name/$($r.name)",'--stop-file',"$linux/results/STOP",'--recovery-variant',$r.variant,'--recovery-world',"$linux/$($r.world)",'--recovery-task',$r.task,'--recovery-split',$r.role,'--recovery-action-budget',[string]$r.action_budget,'--recovery-sampling-seed',[string]$r.sampling_seed)
        if($r.candidate){$argv+=@('--recovery-checkpoint',"$linux/$($r.checkpoint)")}
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($r.name+'-command.json')) -Encoding utf8
        Write-Output "Recovery starting $($r.name)"
        & wsl.exe @argv *> (Join-Path $batch ($r.name+'-console.log'))
        if($LASTEXITCODE -ne 0){throw "Execution failed: $($r.name)"}
        $s=Get-Content -LiteralPath (Join-Path $batch "$($r.name)/summary.json") -Raw | ConvertFrom-Json
        if(-not $s.execution_valid -or $s.worker_exit_code -ne 0 -or $s.process_cleanup.remaining_live_pids.Count -or $s.environment_close_error -or $s.policy_close_error){throw "Invalid execution: $($r.name)"}
        $completed+=@{name=$r.name;steps=$s.steps;completed=$s.program_completed;reason=$s.s2_end_reason;status=$s.status}
        $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output "Recovery finished $($r.name): completed=$($s.program_completed), steps=$($s.steps), reason=$($s.s2_end_reason)"
    }
    Assert-Locked
    & wsl.exe @prefix "$linux/scripts/analyze_rm_general_s2.py" "$linux/results/$name" *> (Join-Path $batch 'raw-audit-console.log')
    if($LASTEXITCODE -ne 0){throw 'Raw performance audit failed'}
    $stopWorld=$development | Where-Object {$_.task -eq 'collection'} | Select-Object -First 1
    $argv=$prefix+@("$linux/scripts/probe_rm_conditioning_stop.py",'--root',$linux,'--batch',$name,'--mode','goal-queue','--program',"$linux/results/$name/programs/stop-wood64.json",'--recovery-variant','nonlinear-on','--recovery-checkpoint',"$linux/results/$name/training/round2/nonlinear-on/checkpoint.pt",'--recovery-world',"$linux/$($stopWorld.world)")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if($LASTEXITCODE -ne 0){throw 'Actual STOP diagnostic failed'}
    foreach($entry in ($locked+$inputs+$official)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Final identity changed: $($entry.path)"}
    }
    @{utc=[DateTime]::UtcNow.ToString('o');completed_runs=$completed.Count;training_jobs=4;stop_passed=$true;official_unchanged=$true;automatic_promotion=$false;API_calls=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} finally {
    if(-not(Test-Path -LiteralPath $stop)){'Common recovery R1 batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
