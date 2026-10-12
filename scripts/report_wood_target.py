"""Assemble correction evidence only after all actual runs and audits finish."""
import datetime,json,re
from pathlib import Path
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1];base=root/'results/wood-target-20261007'
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and len(state['runs'])==3
rows=[]
for item in state['runs']:
    run=Path(item['path']);s=json.loads((run/'summary.json').read_text())
    target=json.loads((run/'wood-target-audit.json').read_text())
    shaping=json.loads((run/'shaping-audit.json').read_text())
    assert target['verified'] and shaping['verified'] and s['execution_valid']
    assert s['rl_updates']==0 and not s['rl_training_eligible'] and not s['process_cleanup']['remaining_live_pids']
    jar=json.loads((run/'runtime-jar-cache.json').read_text())
    assert jar['sha256']=='afeb88a9dd572bad19f25cf8d0a7346c08ef17ab5993e9de5b9da1ebee40688c'
    rows.append(dict(label=item['label'],path=str(run),status=s['status'],steps=s['steps'],
                     elapsed_control_s=s['elapsed_control_s'],inference_ms=s['inference_ms'],
                     control_cycle_ms=s['control_cycle_ms'],gpu=s['gpu'],device_vram=s.get('device_vram'),
                     inventory_final_delta=s['inventory_final_delta'],server_inventory_delta=target['server_inventory_delta'],
                     target=target,shaping=shaping,jar=jar))
for path,sha in {**state['preserved'],**state['source_sha256']}.items():assert digest(root/path)==sha,path
original=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in original['training']['trajectories']:
    p=str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/')
    assert digest(p)==item['sha256']
tests={}
for platform in ('windows','wsl'):
    path=root/f'results/wood-target-tests-{platform}.txt';log=path.read_text()
    assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M)
    tests[platform]=dict(total=int(re.search(r'Ran (\d+) tests',log)[1]),
                        skipped=int(re.search(r'OK \(skipped=(\d+)\)',log)[1]),sha256=digest(path))
assert (root/'results/STOP').read_text().startswith('RL collection ended')
write_json(base/'final-verification.json',dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    runs=rows,tests=tests,initial_comparison=state['initial_comparison'],preserved=state['preserved'],
    original_training_trajectories_verified=len(original['training']['trajectories']),
    optimizer_updates=0,default_policy_promoted=False,final_seeds_unrun=state['final_seeds_unrun']))
print(json.dumps([dict(label=r['label'],steps=r['steps'],status=r['status'],
                      active=r['target']['accepted_server_log_mining_snapshots'],
                      old_stages=r['target']['legacy_stages_during_server_mining']) for r in rows]))
