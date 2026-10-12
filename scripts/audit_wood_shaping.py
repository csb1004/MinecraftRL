"""Recompute shaped rewards from actual server events, without environment actions."""
import argparse,json
from collections import Counter
from pathlib import Path
import torch
from minecraftrl.rl_core import SHAPED_REWARDS,validate_training_data,tick_delta
from minecraftrl.runtime import write_json,jsonable


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    summary=json.loads((a.run/'summary.json').read_text())
    initial=json.loads((a.run/'initial-state.json').read_text())
    data=torch.load(a.run/'trajectory.pt',map_location='cpu')
    if data['split']=='train':validate_training_data(data)
    assert summary['execution_valid'] and summary['worker_exit_code']==0
    assert not summary['process_cleanup']['remaining_live_pids']
    assert summary['release']=='neutral_command_sent' or (
        summary['status']=='environment_done' and summary['release']=='environment_done_no_neutral_step')
    rm=SHAPED_REWARDS[data['reward_config']['name']](initial['server']);index=0;stopped=False;accepted=False;stages=Counter();targets=set();last_transition=None
    sums=dict(task=0.,shaping=0.,time_cost=0.)
    if 'terminal_failure_cost' in data['reward_config']:sums['failure']=0.
    previous=initial['server']
    for line in (a.run/'events.jsonl').open(encoding='utf-8'):
        event=json.loads(line);kind=event['event']
        if kind in ('inference_started','policy_dispatch_started'):assert not stopped and not accepted
        if kind=='stop_acknowledged':stopped=True
        if kind=='target_reached':accepted=True
        if kind!='transition':continue
        last_transition=event
        row=data['rows'][index];index+=1
        assert row['env_action']==event['action'] and row['reward_context']==rm.context()
        # Context is part of both classifier-free guidance branches.
        expected=torch.tensor(row['reward_context'],dtype=row['features'].dtype)
        assert torch.equal(row['features'][0,0,-6:],expected)
        assert torch.equal(row['features'][1,0,-6:],expected)
        server=event['info']['workshop'];tick_delta(previous,server);previous=server
        parts=rm.advance(server,terminal=event['done'])
        assert parts==row['reward_components'] and jsonable(parts)==event['info']['rl_reward']
        assert row['reward']==event['reward']==parts['total']
        stages[parts['stage']]+=1
        prefix='wood_v3_' if data['reward_config']['name'] in ('wood-potential-v3','wood-potential-v4') else 'wood_'
        if server[prefix+'kind']:targets.add(tuple(server[prefix+'target_'+axis] for axis in 'xyz'))
        for k in sums:sums[k]+=parts[k]
        if rm.completed:assert index==summary['steps']
    assert index==summary['steps']==len(data['rows']) and stopped
    assert accepted==rm.completed==summary['server_target_reached']
    assert data['final_reward_context']==rm.context()
    if data['terminal']:assert data['bootstrap']==0 and rm.potential==0
    if summary['status']=='environment_done':
        assert server['server_alive']==0 and server['server_health']==0 and server['held_key_count']==0
        assert last_transition['done'] and not last_transition['observation']['life_stats']['is_alive']
    assert sums==summary['reward_component_sums']
    report=dict(verified=True,steps=index,stages=dict(stages),targets=sorted(targets),
                components=sums,status=summary['status'],success=accepted,
                bootstrap=data['bootstrap'],terminal=data['terminal'],reward_config=data['reward_config'],
                no_policy_action_after_stop=True,one_server_tick_per_action=True,remaining_owned_pids=[],
                scope='actual server transitions and RM wiring; no improvement claim')
    write_json(a.run/'shaping-audit.json',report);print(json.dumps(report))


if __name__=='__main__':main()
