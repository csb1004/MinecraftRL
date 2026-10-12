"""Stream actual transition evidence and verify immutable before/after heads."""
import argparse,json,sys
from pathlib import Path
import torch
from minecraftrl.rl_core import LOGS
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json,distribution

def local(path,root):
    text=str(path).replace('\\','/')
    if text.startswith('D:/DATA/Desktop/MinecraftRL/'):return root/text.split('MinecraftRL/',1)[1]
    return Path(path)

def equal_state(a,b):return a.keys()==b.keys() and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)

def audit_run(folder,expected_actor,expected_value,split):
    r=json.loads((folder/'summary.json').read_text());data=torch.load(folder/'trajectory.pt',map_location='cpu')
    assert data['split']==split and data['eligible'] and len(data['rows'])==r['steps']
    assert equal_state(data['initial_heads']['actor'],expected_actor) and equal_state(data['initial_heads']['value'],expected_value)
    assert r['execution_valid'] and r['release']=='neutral_command_sent' and not r['process_cleanup']['remaining_live_pids']
    acknowledged=False;accepted=False;transitions=0;reward_sum=0;first_server=None;first_client=None
    with (folder/'events.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line)
            if e['event'] in ('inference_started','policy_dispatch_started'):assert not acknowledged and not accepted
            if e['event']=='stop_acknowledged':acknowledged=True
            if e['event']=='target_reached':accepted=True
            if e['event']!='transition':continue
            row=data['rows'][transitions];transitions+=1
            assert row['env_action']==e['action'] and row['reward']==e['reward'] and row['first']==(transitions==1)
            server=sum(e['info']['workshop']['server_inventory'].get(k,0) for k in LOGS)
            client=sum(e['observation']['inventory'].get(k,0) for k in LOGS)
            if server and first_server is None:first_server=transitions
            if client and first_client is None:first_client=transitions
            assert (e['reward']==1)==(first_server==transitions)
            if first_server:assert first_server==r['steps']
            reward_sum+=e['reward']
    assert transitions==r['steps'] and acknowledged and reward_sum==r['reward_sum']
    assert accepted==r['server_target_reached'] and reward_sum==int(accepted)
    assert data['rows'][-1]['terminal']==data['terminal']
    return dict(status=r['status'],success=accepted,steps=r['steps'],seconds=r['elapsed_control_s'],
                reward_sum=reward_sum,first_client_log_step=first_client,first_server_log_step=first_server,
                inference_ms=r['inference_ms'],control_cycle_ms=r['control_cycle_ms'],device_vram=r['device_vram'],
                embedding=data['condition']['embedding_sha256'],initial_frame_sha256=digest(folder/'frames/000000.png'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--train-run',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/steve1'))
    exp=json.loads((a.evaluation/'evaluation.json').read_text());assert exp['completed'] and not exp['unrun']
    training=json.loads((a.training/'training.json').read_text());assert training['completed'] and training['updates']>0
    checkpoint=torch.load(a.training/'checkpoint.pt',map_location='cpu');train=torch.load(a.train_run/'trajectory.pt',map_location='cpu')
    assert digest(a.training/'checkpoint.pt')==exp['checkpoint_sha256']==training['checkpoint_sha256']
    assert digest(a.train_run/'trajectory.pt')==training['trajectory_sha256']
    assert all(state['step'].item()==training['updates'] for state in checkpoint['optimizer']['state'].values())
    results=[]
    train_audit=audit_run(a.train_run,train['initial_heads']['actor'],train['initial_heads']['value'],'train')
    for item in exp['reports']:
        heads=train['initial_heads'] if item['phase']=='before' else checkpoint
        result=audit_run(local(item['path'],root)/'run',heads['actor'],heads['value'],'eval')
        assert result['embedding']==train_audit['embedding']
        results.append(dict(seed=item['seed'],phase=item['phase'],**result))
    paired=[]
    for seed in exp['seeds']:
        pair={r['phase']:r for r in results if r['seed']==seed};assert set(pair)=={'before','after'}
        paired.append(dict(seed=seed,before=pair['before']['success'],after=pair['after']['success'],
                           before_s=pair['before']['seconds'],after_s=pair['after']['seconds'],
                           initial_frame_identical=pair['before']['initial_frame_sha256']==pair['after']['initial_frame_sha256']))
    aggregates={phase:dict(successes=sum(r['success'] for r in results if r['phase']==phase),attempted=5,
        successful_control_seconds=distribution([r['seconds'] for r in results if r['phase']==phase and r['success']])) for phase in ('before','after')}
    report=dict(verified=True,scope='head_only_local_RL_pilot',updates=training['updates'],training=train_audit,
                evaluation=results,paired=paired,aggregates=aggregates,evaluation_used_for_training=False,
                claim='execution of learning and small held-out comparison; not proof of general improvement')
    write_json(a.output,report);print(json.dumps(dict(verified=True,paired=paired,aggregates=aggregates),indent=2))

if __name__=='__main__':main()
