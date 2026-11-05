param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL';$linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='common-recovery-r1-20261101';$batch=Join-Path $root "results/$name";$stop=Join-Path $root 'results/STOP'
if(-not(Test-Path -LiteralPath (Join-Path $batch 'preparation-completion.json'))){throw 'Neutral preparation incomplete'}
if(Test-Path -LiteralPath (Join-Path $batch 'diagnostic-started.json')){throw 'Diagnostic already attempted; preserve evidence'}
if((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Common recovery preparation finished; do not resume automatically.'){throw 'Unrecognized STOP'}
@{utc=[DateTime]::UtcNow.ToString('o');performance_sample=$false;purpose='verify fresh saved fixture restoration and actor dispatch'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'diagnostic-started.json') -Encoding utf8
$env:PYTHONPATH=Join-Path $root 'src'
& (Join-Path $root '.runtime/fixture-tools/Scripts/python.exe') (Join-Path $root 'scripts/prepare_recovery_assets.py') --root $root --batch $batch --mode worlds *> (Join-Path $batch 'world-preparation.log')
if($LASTEXITCODE -ne 0){throw 'Offline worlds preparation failed'}
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linux/src","XDG_CACHE_HOME=$linux/.cache","$linux/.runtime/bin/micromamba",'run','-p',"$linux/.runtime/steve1",'python')
& wsl.exe @prefix "$linux/scripts/prepare_recovery_assets.py" --root $linux --batch "$linux/results/$name" --mode embeddings *> (Join-Path $batch 'embedding-preparation.log')
if($LASTEXITCODE -ne 0){throw 'Goal embedding preparation failed'}
$stopHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
Move-Item -LiteralPath $stop -Destination (Join-Path $batch 'post-preparation-STOP.txt')
@{utc=[DateTime]::UtcNow.ToString('o');prior_stop_sha256=$stopHash;authorization='User approved common recovery and actual verification';performance_sample=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'diagnostic-rearm.json') -Encoding utf8
try {
    $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed','81111001','--limit-s','90','--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/$name/programs/wood1.json",'--output',"$linux/results/$name/diagnostic-game",'--stop-file',"$linux/results/STOP",'--recovery-variant','nonlinear-on','--recovery-world',"$linux/results/$name/diagnostic-world.zip",'--recovery-task','pickup','--recovery-split','development','--recovery-action-budget','400','--recovery-sampling-seed','111099')
    $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'diagnostic-command.json') -Encoding utf8
    Write-Output 'Starting actual restoration/dispatch diagnostic; excluded from train and comparison'
    & wsl.exe @argv *> (Join-Path $batch 'diagnostic-console.log')
    if($LASTEXITCODE -ne 0){throw 'Actual restoration/dispatch diagnostic failed'}
    $s=Get-Content -LiteralPath (Join-Path $batch 'diagnostic-game/summary.json') -Raw | ConvertFrom-Json
    if(-not $s.execution_valid -or $s.worker_exit_code -ne 0 -or $s.process_cleanup.remaining_live_pids.Count){throw 'Invalid diagnostic'}
    @{utc=[DateTime]::UtcNow.ToString('o');execution_valid=$true;steps=$s.steps;goal_completed=$s.program_completed;performance_sample=$false;training_sample=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'diagnostic-completion.json') -Encoding utf8
    Write-Output "Diagnostic ended: steps=$($s.steps), actual pickup=$($s.program_completed)"
} finally {
    if(-not(Test-Path -LiteralPath $stop)){'Common recovery diagnostic finished; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
