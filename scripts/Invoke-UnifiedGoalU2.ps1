param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL';$linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='unified-goal-u2-20261102';$batch=Join-Path $root "results/$name";$stop=Join-Path $root 'results/STOP'
if(Test-Path -LiteralPath (Join-Path $batch 'work.json')){throw 'U2 already claimed; do not rerun'}
$plan=Get-Content -LiteralPath (Join-Path $batch 'plan.json') -Raw | ConvertFrom-Json
if($plan.runs.Count -ne 10){throw 'Expected 10 frozen cases'}
$testLog=Get-Content -LiteralPath (Join-Path $batch 'tests-wsl.log') -Raw
if($testLog -notmatch '(?m)^OK\s*$' -or $testLog -match '(?m)^FAILED'){throw 'Relevant checks have not passed'}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') {throw 'Unrecognized STOP; preserve it'}
$sources=@(foreach($folder in @('src','scripts','tests')) {
    Get-ChildItem -LiteralPath (Join-Path $root $folder) -Recurse -File | Where-Object {$_.Extension -in '.py','.java','.ps1','.sh'}
})
$sources+=Get-Item -LiteralPath (Join-Path $root 'configs/experiment.json'),(Join-Path $root 'configs/logo-wood-validators-v1.json'),(Join-Path $root 'docs/specs/2026-11-02-unified-goal-u2.md')
$locked=@(foreach($file in $sources | Sort-Object FullName -Unique){
    $relative=$file.FullName.Substring($root.Length+1).Replace('\','/')
    $destination=Join-Path $batch "source-at-game/$relative"
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    @{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
})
$inputs=@(foreach($folder in @('goal-bank','programs')){
    foreach($file in Get-ChildItem -LiteralPath (Join-Path $batch $folder) -Recurse -File){
        @{path=$file.FullName.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
    }
})
$inputs+=@{path=$plan.runs[0].checkpoint;sha256=(Get-FileHash -LiteralPath (Join-Path $root $plan.runs[0].checkpoint) -Algorithm SHA256).Hash}
$inputs+=@{path="results/$name/plan.json";sha256=(Get-FileHash -LiteralPath (Join-Path $batch 'plan.json') -Algorithm SHA256).Hash}
$official=@(foreach($path in @('weights/steve1/steve1.weights','weights/steve1/steve1_prior.pt','weights/steve1/attn.pth','weights/steve1/2x.model','docs/research-spec.md')){
    @{path=$path;sha256=(Get-FileHash -LiteralPath (Join-Path $root $path) -Algorithm SHA256).Hash}
})
@{locked=$locked;inputs=$inputs;official=$official;git_head=(git -C $root rev-parse HEAD);source_snapshot='source-at-game';modified_worktree=$true} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
function Assert-Locked {
    if(Test-Path -LiteralPath $stop){throw 'STOP latched; do not start next run'}
    foreach($entry in ($locked+$inputs)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Frozen input changed: $($entry.path)"}
    }
}
$prior=Join-Path $batch 'prior-STOP.txt'
$stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linux/src","XDG_CACHE_HOME=$linux/.cache","$linux/.runtime/bin/micromamba",'run','-p',"$linux/.runtime/steve1",'python')
$completed=@()
try {
    Move-Item -LiteralPath $stop -Destination $prior
    if((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $stopHash){throw 'STOP changed during claim'}
    @{utc=[DateTime]::UtcNow.ToString('o');prior_stop_sha256=$stopHash;authorization='User requested actual unified-goal experiment and bottleneck analysis on 2026-11-02';claimed_once=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
    foreach($r in $plan.runs){
        Assert-Locked
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$r.seed,'--limit-s','180','--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/$($r.program)",'--output',"$linux/results/$name/$($r.name)",'--stop-file',"$linux/results/STOP",'--s2-variant','linear-on','--s2-checkpoint',"$linux/$($r.checkpoint)",'--s2-split','development','--s2-action-budget','1200')
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($r.name+'-command.json')) -Encoding utf8
        Write-Output "U2 starting $($r.name)"
        & wsl.exe @argv *> (Join-Path $batch ($r.name+'-console.log'))
        if($LASTEXITCODE -ne 0){throw "Execution failed: $($r.name)"}
        $s=Get-Content -LiteralPath (Join-Path $batch "$($r.name)/summary.json") -Raw | ConvertFrom-Json
        if(-not $s.execution_valid -or $s.worker_exit_code -ne 0 -or $s.process_cleanup.remaining_live_pids.Count -or $s.environment_close_error -or $s.policy_close_error){throw "Invalid execution: $($r.name)"}
        $completed+=@{name=$r.name;steps=$s.steps;completed=$s.program_completed;reason=$s.s2_end_reason;status=$s.status}
        $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $batch 'driver-progress.json') -Encoding utf8
        Write-Output "U2 finished $($r.name): completed=$($s.program_completed), steps=$($s.steps), reason=$($s.s2_end_reason)"
    }
    Assert-Locked
    & wsl.exe @prefix "$linux/scripts/analyze_rm_general_s2.py" "$linux/results/$name" *> (Join-Path $batch 'raw-audit-console.log')
    if($LASTEXITCODE -ne 0){throw 'Raw performance audit failed'}
    $argv=$prefix+@("$linux/scripts/probe_rm_conditioning_stop.py",'--root',$linux,'--batch',$name,'--mode','goal-queue','--program',"$linux/results/$name/programs/stop-wood64.json",'--s2-variant','linear-on','--s2-checkpoint',"$linux/$($plan.runs[0].checkpoint)")
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'stop-command.json') -Encoding utf8
    & wsl.exe @argv *> (Join-Path $batch 'stop-helper.log')
    if($LASTEXITCODE -ne 0){throw 'Actual STOP diagnostic failed'}
    foreach($entry in ($locked+$inputs+$official)){
        if((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256){throw "Final identity changed: $($entry.path)"}
    }
    @{utc=[DateTime]::UtcNow.ToString('o');completed_runs=$completed.Count;training_jobs=0;stop_passed=$true;official_unchanged=$true;automatic_promotion=$false;API_calls=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
} finally {
    if(-not(Test-Path -LiteralPath $stop)){'U2 batch ended; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
