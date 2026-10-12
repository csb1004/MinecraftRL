"""Read-only transition audit and descriptive comparison of fixed-checkpoint trials."""
import json,sys
from pathlib import Path
from collections import defaultdict
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_fixture_probe import audit as audit_probe
from audit_wood_shaping import main as audit_policy

ROOT=Path(__file__).resolve().parents[1]
def pressed(value):
    while isinstance(value,list):
        if len(value)!=1:raise ValueError('expected scalar button')
        value=value[0]
    return bool(value)

def local(path):
    normalized=str(path).replace('\\','/')
    if normalized.startswith('D:/DATA/Desktop/MinecraftRL/'):
        return ROOT/normalized.split('MinecraftRL/',1)[1]
    return Path(path)

def main():
    base=ROOT/'results/wood-followup-20261007'
    state=json.loads((base/'experiment.json').read_text())
    rows=[]
    for item in state['runs']:
        run=local(item['path']);r=json.loads((run/'summary.json').read_text())
        if not r['execution_valid']:
            stopped=False;dispatches=0;transitions=0
            for line in (run/'events.jsonl').read_text().splitlines():
                e=json.loads(line)
                if e['event']=='stop_acknowledged':stopped=True
                if e['event'] in ('inference_started','policy_dispatch_started'):assert not stopped
                dispatches+=e['event']=='policy_dispatch_started'
                transitions+=e['event']=='transition'
            assert stopped and not r['process_cleanup']['remaining_live_pids']
            assert not r.get('rl_training_eligible',False)
            rows.append(dict(label=item['label'],valid=False,status=r['status'],error=r.get('error'),
                attempted_dispatches=dispatches,logged_transitions=transitions,no_action_after_stop=True,
                release=r['release'],remaining_owned_pids=[]));continue
        if 'probe' in item: audit_probe(run)
        else:
            sys.argv=['audit_wood_shaping.py',str(run)];audit_policy()
        row=dict(label=item['label'],valid=True,status=r['status'],success=r['server_target_reached'],
            steps=r['steps'],seconds=r['elapsed_control_s'],inference_ms=r['inference_ms'],
            control_cycle_ms=r['control_cycle_ms'],gpu=r['gpu'],device_vram=r.get('device_vram'),
            inventory_gain=r['inventory_final_delta'],scripted='probe' in item,
            path=str(run.relative_to(ROOT)))
        initial=json.loads((run/'initial-state.json').read_text())['server']['server_inventory']
        last_server=None
        for line in (run/'events.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e['event']=='transition':last_server=e['info']['workshop']['server_inventory']
        row['server_inventory_delta']={k:v-initial.get(k,0) for k,v in last_server.items() if v!=initial.get(k,0)}
        if item['label'].startswith(('probe-break','probe-detour','policy-')):
            manifest=json.loads(local(item['world']).with_suffix('.json').read_text())
            ox,oz=[v*16 for v in manifest['chunk']]
            crossing=None;tree_attacks=0;tree_attacks_context_elsewhere=0
            for line in (run/'events.jsonl').read_text().splitlines():
                e=json.loads(line)
                if e['event']!='transition':continue
                g=e['info']['workshop']
                if crossing is None and g['server_x']>=ox+7.5:
                    crossing=dict(x=g['server_x']-ox,y=g['server_y'],z=g['server_z']-oz)
                if pressed(e['action'].get('attack',0)) and g['hit_x']==ox+11 and g['hit_z']==oz+3 and 201<=g['hit_y']<=204:
                    tree_attacks+=1
                    if tuple(g['hit_'+k] for k in 'xyz')!=tuple(g['wood_target_'+k] for k in 'xyz'):
                        tree_attacks_context_elsewhere+=1
            row['first_barrier_crossing']=crossing
            row['tree_attack_hit_samples']=tree_attacks
            row['tree_attack_hit_differs_from_locked_target']=tree_attacks_context_elsewhere
        rows.append(row)
    preserved=all(digest(local(p))==sha for p,sha in state['world_sha256'].items())
    preserved=preserved and all(digest(local(v['path']))==v['sha256'] for v in state['checkpoints'].values())
    assert preserved
    natural=defaultdict(list)
    for row in rows:
        if row['label'].startswith('natural-') and row['valid']:natural[row['label'].split('-')[-1]].append(row)
    stats={k:dict(n=len(v),successes=sum(r['success'] for r in v),
        mean_capped_seconds=sum(r['seconds'] if r['success'] else 180 for r in v)/len(v)) for k,v in natural.items()}
    result=dict(audited=True,batch_completed=state['completed'],rows=rows,natural=stats,
        initial_comparisons=state.get('comparisons',{}),sources_preserved=preserved,
        unrun=state['unrun'],errors=state['errors'],optimizer_updates=0,default_policy_promoted=False)
    write_json(base/'audit.json',result)
    print(json.dumps(dict(audited=len(rows),natural=stats,unrun=len(state['unrun']))))

if __name__=='__main__':main()
