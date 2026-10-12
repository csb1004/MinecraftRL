"""Audit explicitly scripted diagnostics; these are never learned-policy samples."""
import argparse,json
from pathlib import Path
from collections import Counter
from minecraftrl.rl_core import SHAPED_REWARDS,tick_delta
from minecraftrl.runtime import write_json

def audit(run):
    report=json.loads((run/'summary.json').read_text())
    initial=json.loads((run/'initial-state.json').read_text())
    rows=json.loads((run/'probe-rows.json').read_text())
    assert report['controller']=='scripted_fixture_probe'
    assert not report['rl_training_eligible'] and not report['learned_policy_loaded']
    assert report['execution_valid'] and not report['process_cleanup']['remaining_live_pids']
    rm=SHAPED_REWARDS['wood-potential-v2'](initial['server'])
    stopped=accepted=False;count=0;stages=Counter();failure=0.;last=None;previous=initial['server']
    for line in (run/'events.jsonl').read_text().splitlines():
        e=json.loads(line);kind=e['event']
        if kind in ('inference_started','policy_dispatch_started'):assert not stopped and not accepted
        if kind=='stop_acknowledged':stopped=True
        if kind=='target_reached':accepted=True
        if kind!='transition':continue
        row=rows[count];count+=1
        assert row['action']==e['action']
        last=e;server=e['info']['workshop'];tick_delta(previous,server);previous=server
        parts=rm.advance(server,terminal=e['done'])
        assert row['terminal']==bool(e['done'] or rm.completed)
        assert parts==e['info']['rl_reward'] and parts['total']==e['reward']==row['reward']
        failure+=parts['failure'];stages[parts['stage']]+=1
        if rm.completed:assert count==report['steps']
    assert stopped and count==report['steps']==len(rows)
    assert accepted==rm.completed==report['server_target_reached']
    if report['status']=='environment_done':
        assert last['done'] and report['release'] in ('neutral_command_sent','environment_done_no_neutral_step')
    else:assert report['release']=='neutral_command_sent'
    if report['route']=='death':
        server=last['info']['workshop']
        assert report['status']=='environment_done' and failure==-1.
        assert server['server_alive']==0 and server['server_health']==0 and server['held_key_count']==0
        assert not last['observation']['life_stats']['is_alive']
    result=dict(verified=True,controller=report['controller'],status=report['status'],steps=count,
                success=accepted,stages=dict(stages),failure_cost=failure,
                no_action_after_stop=True,one_server_tick_per_action=True,remaining_owned_pids=[],training_eligible=False)
    if report['route']=='death':result['terminal_server_evidence']={k:server[k] for k in ('server_health','server_alive','held_key_count')}
    write_json(run/'probe-audit.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    print(json.dumps(audit(p.parse_args().run)))
