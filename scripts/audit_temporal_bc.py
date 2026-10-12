"""Audit actual loaded weights, server outcomes, motion and STOP for the bounded comparison."""
import argparse,json
from pathlib import Path
import torch
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

parser=argparse.ArgumentParser();parser.add_argument('--diagnostic-kl',action='store_true');parser.add_argument('--base',type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
base=root/args.base if args.base else root/('results/temporal-bc-diagnostic-20261008' if args.diagnostic_kl else 'results/temporal-bc-comparison-20261008')
if not base.resolve().is_relative_to(root/'results'):raise ValueError('project-local comparison required')
out=base/'audit.json'
if out.exists():raise FileExistsError(out)
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and state['active'] is None

def local(path):return root/str(path).replace('\\','/').split('/MinecraftRL/',1)[-1]

audits=[]
for trial in state['runs']:
    run=local(trial['path']);report=json.loads((run/'summary.json').read_text())
    checkpoint=local(state['parent' if trial['label'].endswith('-parent') else 'candidate'])
    assert trial['audited'] and digest(checkpoint)==trial['checkpoint_sha256']
    assert report['execution_valid'] and report['worker_exit_code']==0 and not report['rl_training_eligible']
    assert not report['process_cleanup']['remaining_live_pids']
    release=json.loads((run/'input-release-state.json').read_text())
    assert release['verified'] and release['held_key_count']==0
    saved=torch.load(checkpoint,map_location='cpu');data=torch.load(run/'trajectory.pt',map_location='cpu')
    assert data['split']=='eval'
    for group in ('actor','value','backbone_tail'):
        assert state_fingerprint(saved[group])==state_fingerprint(data['initial_heads'][group])
    if saved['scope']=='head_tail_last_recurrent':
        assert state_fingerprint(saved['backbone_temporal'])==state_fingerprint(data['initial_heads']['backbone_temporal'])
        assert data['condition']['backbone_temporal_sha256']==state_fingerprint(saved['backbone_temporal'])
    else:assert 'backbone_temporal' not in data['initial_heads']
    del saved,data
    previous=json.loads((run/'initial-state.json').read_text())['server'];initial=previous
    distance=turn=0.;moved=rotated=count=0;stopped=accepted=False
    keys={k:0 for k in ('forward','back','left','right','jump','sprint','sneak','attack','use','inventory')}
    for line in (run/'events.jsonl').open():
        event=json.loads(line)
        if event['event'] in ('inference_started','policy_dispatch_started'):assert not stopped and not accepted
        if event['event']=='stop_acknowledged':stopped=True
        if event['event']=='target_reached':accepted=True
        if event['event']!='transition':continue
        current=event['info']['workshop'];count+=1
        for key in keys:
            value=event['action'].get(key,0)
            keys[key]+=int(bool(value[0] if isinstance(value,list) else value))
        step=sum((current['server_'+axis]-previous['server_'+axis])**2 for axis in 'xz')**.5
        rotation=abs((current['yaw']-previous['yaw']+180)%360-180)+abs(current['pitch']-previous['pitch'])
        distance+=step;turn+=rotation;moved+=step>1e-5;rotated+=rotation>1e-5;previous=current
    assert stopped and count==report['steps']
    delta={k:v-initial['server_inventory'].get(k,0) for k,v in previous['server_inventory'].items()
           if v!=initial['server_inventory'].get(k,0)}
    success=any(delta.get(k+'_log',0)>0 for k in ('oak','birch','spruce','jungle','acacia','dark_oak'))
    assert success==report['server_target_reached']
    audits.append(dict(label=trial['label'],run=trial['path'],steps=count,server_success=success,inventory_delta=delta,
                       checkpoint_verified=True,commands=keys,horizontal_path_blocks=distance,moving_transitions=moved,
                       turning_transitions=rotated,absolute_yaw_pitch_change_degrees=turn,
                       elapsed_control_s=report['elapsed_control_s'],inference_ms=report['inference_ms'],
                       control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
                       post_stop_policy_dispatches=0,released_held_keys=0,remaining_owned_pids=[],
                       source_sha256={name:digest(run/name) for name in ('events.jsonl','summary.json','trajectory.pt','initial-state.json')}))
write_json(out,dict(verified=True,rows=audits,unrun_conditions=4-len(audits),comparison_valid=state.get('comparison_valid',{}),
                    gate_policy=state['gate_policy'],strict_gate_passed=state['strict_gate_passed'],
                    scope='Actual server outcomes and loaded BC weights; development comparison, not final evaluation'))
print(json.dumps(audits))
