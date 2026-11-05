param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root='D:\DATA\Desktop\MinecraftRL'
$linux='/mnt/d/DATA/Desktop/MinecraftRL'
$name='common-recovery-r1-20261101'
$batch=Join-Path $root "results/$name"
$stop=Join-Path $root 'results/STOP'
if (Test-Path -LiteralPath (Join-Path $batch 'preparation-started.json')) {throw 'Preparation already claimed; inspect records instead of rerunning'}
New-Item -ItemType Directory -Path $batch -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'results/rm-general-s2-20261101/goal-bank') -Destination (Join-Path $batch 'goal-bank') -Recurse
New-Item -ItemType Directory -Path (Join-Path $batch 'programs') | Out-Null
foreach($key in @('wood1','dirt1')) {
    Copy-Item -LiteralPath (Join-Path $root "results/rm-general-s2-20261101/programs/$key.json") -Destination (Join-Path $batch "programs/$key.json")
}
if ((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') {throw 'Unrecognized STOP; preserve it'}
$hash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
Move-Item -LiteralPath $stop -Destination (Join-Path $batch 'prior-STOP.txt')
if ((Get-FileHash -LiteralPath (Join-Path $batch 'prior-STOP.txt') -Algorithm SHA256).Hash -ne $hash) {
    if(-not(Test-Path -LiteralPath $stop)){Copy-Item -LiteralPath (Join-Path $batch 'prior-STOP.txt') -Destination $stop}
    throw 'STOP changed during claim'
}
@{utc=[DateTime]::UtcNow.ToString('o');authorization='User approved common recovery improvement including saved starts and actual tests';prior_stop_sha256=$hash;policy_actions=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'preparation-started.json') -Encoding utf8
$prefix=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',"PYTHONPATH=$linux/src","XDG_CACHE_HOME=$linux/.cache","$linux/.runtime/bin/micromamba",'run','-p',"$linux/.runtime/steve1",'python')
$seeds=@(81111001,81112001,81112003,81112007,81112009,81112013)
try {
    foreach($seed in $seeds) {
        if(Test-Path -LiteralPath $stop){throw 'STOP; no further preparation'}
        $id="prepare-$seed"
        $argv=$prefix+@('-m','minecraftrl.cli','episode','--root',$linux,'--policy','steve1','--condition','wood','--seed',[string]$seed,'--limit-s','180','--rm-conditioning','goal-queue','--rm-goal-bank',"$linux/results/$name/goal-bank",'--rm-program',"$linux/results/$name/programs/wood1.json",'--recovery-prepare','--output',"$linux/results/$name/$id",'--stop-file',"$linux/results/STOP")
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch "$id-command.json") -Encoding utf8
        Write-Output "Preparing neutral saved start $seed"
        & wsl.exe @argv *> (Join-Path $batch "$id-console.log")
        if($LASTEXITCODE -ne 0){throw "Preparation failed: $seed"}
        $summary=Get-Content -LiteralPath (Join-Path $batch "$id/summary.json") -Raw | ConvertFrom-Json
        if(-not $summary.execution_valid -or $summary.status -ne 'world_prepared' -or $summary.steps -ne 0 -or $summary.process_cleanup.remaining_live_pids.Count){throw "Invalid preparation: $seed"}
        Write-Output "Saved start $seed complete; policy actions zero"
    }
    @{utc=[DateTime]::UtcNow.ToString('o');seeds=$seeds;policy_actions=0;performance_samples=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'preparation-completion.json') -Encoding utf8
} finally {
    if(-not (Test-Path -LiteralPath $stop)) {'Common recovery preparation finished; do not resume automatically.' | Set-Content -LiteralPath $stop -Encoding utf8}
}
