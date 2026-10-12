"""Replay the recovery detector on actual snapshots, separate entry from success."""
import argparse,json
from pathlib import Path
from minecraftrl.approach_recovery import ApproachRecovery
from minecraftrl.rl_core import ShapedWoodRewardV3,tick_delta
from minecraftrl.runtime import write_json,jsonable

def audit(run):
    summary=json.loads((run/'summary.json').read_text());initial=json.loads((run/'initial-state.json').read_text())['server']
    detector=ApproachRecovery();detector.update(initial,ShapedWoodRewardV3.state(initial)[0])
    previous=initial;events=[];active=0;steps=0;stopped=accepted=False
    for line in (run/'events.jsonl').open():
        e=json.loads(line)
        if e['event'] in ('inference_started','policy_dispatch_started'):assert not stopped and not accepted
        if e['event']=='stop_acknowledged':stopped=True
        if e['event']=='target_reached':accepted=True
        if e['event']!='transition':continue
        s=e['info']['workshop'];steps+=1;tick_delta(previous,s);previous=s
        accepting=any(s['server_inventory'].get(k+'_log',0)>0 for k in ('oak','birch','spruce','jungle','acacia','dark_oak'))
        r=detector.update(s,ShapedWoodRewardV3.state(s)[0],terminal=e['done'] or accepting)
        if summary['reward_config']['name']=='wood-potential-v4':
            assert jsonable(r)==e['info']['rl_reward']['recovery']
            if r['active']:assert e['info']['rl_reward']['stage']=='search'
        if r['event']:events.append(dict(step=steps,**r))
        active+=int(r['active'])
    assert summary['execution_valid'] and summary['worker_exit_code']==0 and stopped
    assert not summary['process_cleanup']['remaining_live_pids'] and steps==summary['steps']
    entered=any(e['event']=='entered' for e in events)
    result=dict(verified=True,mode=summary['reward_config']['name'],steps=steps,events=events,active_snapshots=active,
                success=summary['server_target_reached'],entered=entered,
                acquisition_after_entry=entered and summary['server_target_reached'],
                server_inventory_delta={k:v-initial['server_inventory'].get(k,0) for k,v in previous['server_inventory'].items() if v!=initial['server_inventory'].get(k,0)},
                last_server={k:previous[k] for k in ('server_health','server_alive','held_key_count')},
                scope='v3 detector is counterfactual; v4 detector is wired; success is actual server acquisition')
    write_json(run/'recovery-audit.json',result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);audit(p.parse_args().run)
