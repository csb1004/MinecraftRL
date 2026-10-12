"""Collect actual outcomes without turning detector entry into task success."""
import datetime,json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/approach-recovery-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and len(state['runs'])==4
rows=[]
for item in state['runs']:
    run=Path(item['path']);s=json.loads((run/'summary.json').read_text());probe='probe' in item
    check=json.loads((run/('probe-audit.json' if probe else 'shaping-audit.json')).read_text())
    assert check['verified'] and s['execution_valid'] and s['worker_exit_code']==0
    assert not s['rl_training_eligible'] and not s['process_cleanup']['remaining_live_pids']
    jar=json.loads((run/'runtime-jar-cache.json').read_text())
    assert jar['sha256']=='afeb88a9dd572bad19f25cf8d0a7346c08ef17ab5993e9de5b9da1ebee40688c'
    extra={} if probe else json.loads((run/'recovery-audit.json').read_text())
    rows.append(dict(label=item['label'],path=str(run),scripted=probe,status=s['status'],steps=s['steps'],
                     seconds=s['elapsed_control_s'],inference_ms=s['inference_ms'],control_cycle_ms=s['control_cycle_ms'],
                     gpu=s['gpu'],device_vram=s.get('device_vram'),audit=check,recovery=extra,
                     server_success=s['server_target_reached']))
for path,sha in {**state['source_sha256'],**state['preserved']}.items():assert digest(root/path)==sha,path
old=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in old['training']['trajectories']:
    assert digest(str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/'))==item['sha256']
tests={}
for platform in ('windows','wsl','fixture'):
    path=root/f'results/approach-recovery-tests-{platform}.txt';text=path.read_text()
    assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',text,re.M)
    skipped=re.search(r'OK \(skipped=(\d+)\)',text)
    tests[platform]=dict(total=int(re.search(r'Ran (\d+) tests',text)[1]),skipped=int(skipped[1]) if skipped else 0,sha256=digest(path))
assert (root/'results/STOP').read_text().startswith('RL collection ended')
context=json.loads((base/'context-intervention.json').read_text())
result=dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=rows,
            tests=tests,initial_comparison=state['initial_comparison'],context_intervention=context,
            preserved=state['preserved'],original_training_trajectories_verified=len(old['training']['trajectories']),
            optimizer_updates=0,default_policy_promoted=False,final_seeds_unrun=state['final_seeds_unrun'])
write_json(base/'final-verification.json',result)
print(json.dumps([dict(label=r['label'],steps=r['steps'],status=r['status'],
                      entered=r['recovery'].get('entered'),acquisition_after_entry=r['recovery'].get('acquisition_after_entry')) for r in rows]))
