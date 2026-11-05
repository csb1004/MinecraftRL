[CmdletBinding()]
param([string]$Distribution='Ubuntu-24.04')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$batch=Join-Path $root 'results/rm-logo-conditioning-20261031'
$stop=Join-Path $root 'results/STOP'
$protocol=Join-Path $root 'docs/specs/2026-10-31-logo-conditioning-p3.md'
if (Test-Path -LiteralPath (Join-Path $batch 'pilot-plan.json')) { throw 'P3 already started; no automatic rerun.' }
New-Item -ItemType Directory -Path $batch -Force | Out-Null
$linuxRoot=(& wsl.exe -d $Distribution -- wslpath -a $root.Replace('\','/')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'WSL mapping failed' }
foreach ($pair in @(@('results/logo-validators-20261031/goal-bank','goal-bank'),@('results/rm-purpose-conditioning-20261031/goal-bank','purpose-bank'))) {
    $destination=Join-Path $batch $pair[1]
    if (Test-Path -LiteralPath $destination) { throw 'Prepared bank destination already exists; inspect evidence.' }
    Copy-Item -LiteralPath (Join-Path $root $pair[0]) -Destination $destination -Recurse
}
$seedList=@(81031001,81031003,81031007,81031009,81031013)
$modes=@('logo-staged','purpose-fixed','logo-fixed')
$plan=@()
for ($i=0; $i -lt $seedList.Count; $i++) {
    for ($j=0; $j -lt 3; $j++) {
        $mode=$modes[($i+$j)%3]
        $bank=if ($mode -eq 'purpose-fixed') { 'purpose-bank' } else { 'goal-bank' }
        $plan+=@{seed=$seedList[$i];mode=$mode;bank=$bank;name=([string]$seedList[$i]+'-'+$mode)}
    }
}
$snapshot=Join-Path $batch 'source-at-game'
New-Item -ItemType Directory -Path $snapshot | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'src') -Destination $snapshot -Recurse
foreach ($path in @('configs/experiment.json','configs/logo-wood-validators-v1.json','scripts/probe_rm_conditioning_stop.py')) {
    Copy-Item -LiteralPath (Join-Path $root $path) -Destination $snapshot
}
Copy-Item -LiteralPath $protocol -Destination $snapshot
Copy-Item -LiteralPath $PSCommandPath -Destination $snapshot
$locked=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/minecraftrl') -File | Where-Object { $_.Extension -in '.py','.java' } | ForEach-Object { $_.FullName })
$locked+=@($protocol,$PSCommandPath,(Join-Path $root 'configs/experiment.json'),(Join-Path $root 'scripts/probe_rm_conditioning_stop.py'))
$locked+=@(Get-ChildItem -LiteralPath (Join-Path $root 'weights/steve1') -File -Filter '*.receipt.json' | ForEach-Object { $_.FullName })
foreach ($bank in @('goal-bank','purpose-bank')) {
    $locked+=@(Get-ChildItem -LiteralPath (Join-Path $batch $bank) -File | Where-Object { $_.Name -eq 'manifest.json' -or $_.Extension -eq '.npy' } | ForEach-Object { $_.FullName })
}
$hashes=@($locked | ForEach-Object { @{path=$_.Substring($root.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
@{schema=1;protocol='P3';seeds=$seedList;runs=$plan;limit_s=180;split='development';optimizer_updates=0;
  protocol_sha256=(Get-FileHash -LiteralPath $protocol -Algorithm SHA256).Hash} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $batch 'pilot-plan.json') -Encoding utf8
@{git_head=(git -C $root rev-parse HEAD);locked=$hashes;modified_worktree=$true;source_snapshot='source-at-game';
  research_spec_sha256=(Get-FileHash -LiteralPath (Join-Path $root 'docs/research-spec.md') -Algorithm SHA256).Hash} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batch 'work.json') -Encoding utf8
# This user turn authorizes a new experiment. Preserve and atomically claim only
# the known previous diagnostic STOP; never remove a concurrent new user STOP.
if (Test-Path -LiteralPath $stop) {
    if ((Get-Content -LiteralPath $stop -Raw).Trim() -ne 'Actual-game STOP probe; do not resume automatically.') {
        throw 'Different STOP message; retain it and inspect the user instruction.'
    }
    $priorHash=(Get-FileHash -LiteralPath $stop -Algorithm SHA256).Hash
    $prior=Join-Path $batch 'prior-STOP.txt'
    Move-Item -LiteralPath $stop -Destination $prior
    if ((Get-FileHash -LiteralPath $prior -Algorithm SHA256).Hash -ne $priorHash) {
        if (-not (Test-Path -LiteralPath $stop)) { Copy-Item -LiteralPath $prior -Destination $stop }
        throw 'STOP changed during claim; execution remains cancelled.'
    }
    @{authorization='User requested experiment in current turn (2026-10-31)';prior_stop_sha256=$priorHash;
      claimed_once=$true;utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'rearm-receipt.json') -Encoding utf8
}
$completed=0
try {
    foreach ($run in $plan) {
        if (Test-Path -LiteralPath $stop) { throw 'STOP observed; remaining conditions unexecuted.' }
        foreach ($entry in $hashes) {
            if ((Get-FileHash -LiteralPath (Join-Path $root $entry.path) -Algorithm SHA256).Hash -ne $entry.sha256) {
                throw ('Locked artifact changed: '+$entry.path)
            }
        }
        $argv=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',
            "PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache",
            "$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python','-m','minecraftrl.cli',
            'episode','--root',$linuxRoot,'--policy','steve1','--condition','wood','--seed',([string]$run.seed),
            '--limit-s','180','--rm-conditioning',$run.mode,'--rm-goal-bank',"$linuxRoot/results/rm-logo-conditioning-20261031/$($run.bank)",
            '--output',"$linuxRoot/results/rm-logo-conditioning-20261031/$($run.name)",'--stop-file',"$linuxRoot/results/STOP")
        $argv | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch ($run.name+'-command.json')) -Encoding utf8
        Write-Output ('Starting '+$run.name+' '+[DateTime]::Now.ToString('o'))
        & wsl.exe @argv *> (Join-Path $batch ($run.name+'-console.log'))
        $runExit=$LASTEXITCODE
        $summaryPath=Join-Path $batch ($run.name+'/summary.json')
        if (-not (Test-Path -LiteralPath $summaryPath)) { throw ('No supervisor summary: '+$run.name+' exit '+$runExit) }
        $summary=Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        Write-Output ($run.name+': '+$summary.status+' steps='+$summary.steps+' server_goal='+$summary.server_target_reached)
        if ($runExit -ne 0 -or -not $summary.execution_valid) { throw 'Technical error/cancellation; preserve evidence and leave remaining runs unexecuted.' }
        $completed++
    }
    if (Test-Path -LiteralPath $stop) { throw 'STOP observed before separate stop diagnostic.' }
    $probe=@('-d',$Distribution,'--','env','LD_LIBRARY_PATH=/usr/lib/wsl/lib','GALLIUM_DRIVER=d3d12',
        "PYTHONPATH=$linuxRoot/src","XDG_CACHE_HOME=$linuxRoot/.cache",
        "$linuxRoot/.runtime/bin/micromamba",'run','-p',"$linuxRoot/.runtime/steve1",'python',
        "$linuxRoot/scripts/probe_rm_conditioning_stop.py",'--root',$linuxRoot,'--batch','rm-logo-conditioning-20261031','--mode','logo-staged')
    Write-Output 'Performance batch finished; starting separate actual-game STOP diagnostic.'
    & wsl.exe @probe *> (Join-Path $batch 'stop-helper.log')
    if ($LASTEXITCODE -ne 0) { throw 'STOP diagnostic error; inspect preserved evidence.' }
    @{performance_runs=$completed;stop_probe_passed=$true;utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batch 'driver-completion.json') -Encoding utf8
}
finally {
    if (-not (Test-Path -LiteralPath $stop)) {
        try {
            $stream=[System.IO.File]::Open($stop,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
            $writer=[System.IO.StreamWriter]::new($stream)
            $writer.WriteLine('P3 driver ended; controlled re-arm required.')
            $writer.Dispose()
        } catch [System.IO.IOException] { }
    }
    Write-Output ('P3 driver ended; completed performance conditions='+$completed+'. STOP retained.')
}
