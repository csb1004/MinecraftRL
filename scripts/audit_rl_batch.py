"""Read-only audit of the actual multi-seed batch and its simulation-time evidence."""
import argparse,json,sys
from pathlib import Path
import torch
from audit_rl_pilot import audit_run,local
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json
from minecraftrl.rl_core import LOGS

def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/steve1'))
    state=json.loads((a.batch/'batch.json').read_text());assert state['completed'] and not state['errors']
    previous=torch.load(root/'results/rl-head-training-20261005/checkpoint.pt',map_location='cpu')
    training_folder=local(state.get('training_directory',a.batch/'training'),root)
    updated=torch.load(training_folder/'checkpoint.pt',map_location='cpu');training=state['training']
    pilot=json.loads((root/'results/rl-head-training-20261005/training.json').read_text())
    assert digest(root/'results/rl-head-training-20261005/checkpoint.pt')==training['parent_checkpoint_sha256']==pilot['checkpoint_sha256']
    assert digest(training_folder/'checkpoint.pt')==training['checkpoint_sha256']
    assert training['total_updates']==previous['updates']+training['updates']==updated['updates']
    assert all(v['step'].item()==updated['updates'] for v in updated['optimizer']['state'].values())
    assert set(state['train_seeds']).isdisjoint(state['validation_seeds']+state['reserved_final_seeds'])
    assert state['reproducibility']['state_identical'] and state['reproducibility']['pixel_identical']
    assert all(v['state_identical'] and v['pixel_identical'] for v in state['validation_initial'].values())
    trained_paths={local(v['path'],root).resolve():v for v in training['trajectories']}
    results=[]
    for run in state['runs']:
        if run['label'].startswith('prepare-'):continue
        folder=local(run['path'],root);r=json.loads((folder/'summary.json').read_text())
        is_train=run['label'].startswith('train-');heads=updated if run['label'].endswith('-after') else previous
        if is_train and r['status']=='time_limit':assert r['condition'].get('recurrent_commit_mode')=='dispatch_commit'
        result=audit_run(folder,heads['actor'],heads['value'],'train' if is_train else 'eval')
        trajectory=(folder/'trajectory.pt').resolve()
        assert (trajectory in trained_paths)==is_train
        if is_train:assert digest(trajectory)==trained_paths[trajectory]['sha256']
        initial=json.loads((folder/'initial-state.json').read_text());tick=initial['server']['server_tick'];world_time=initial['server']['world_time'];deltas=[]
        assert not any(initial['server']['server_inventory'].values()) and r['neutral_startup_steps']==60
        with (folder/'events.jsonl').open() as stream:
            for line in stream:
                e=json.loads(line)
                if e['event']!='transition':continue
                server=e['info']['workshop'];delta=server['server_tick']-tick
                assert delta==1 and server['world_time']-world_time==1
                deltas.append(delta);tick=server['server_tick'];world_time=server['world_time']
        assert deltas==r['server_tick_deltas'] and sum(deltas)==r['simulation_ticks']==r['steps']
        final_inventory={k:int(v) for k,v in server['server_inventory'].items() if v}
        results.append(dict(label=run['label'],simulation_ticks=r['simulation_ticks'],nominal_simulation_seconds=r['nominal_simulation_seconds'],
                            final_server_inventory=final_inventory,final_server_logs=sum(final_inventory.get(k,0) for k in LOGS),**result))
    assert len(trained_paths)==3
    for seed in state['validation_seeds']:
        pair=[r for r in results if r['label'].startswith('validation-'+str(seed)+'-')]
        assert len(pair)==2 and pair[0]['initial_frame_sha256']==pair[1]['initial_frame_sha256']
    for item in state['worlds'].values():assert digest(local(item['path'],root))==item['sha256']
    report=dict(verified=True,updates=training['updates'],total_updates=training['total_updates'],
                runs=results,training_samples=training['samples'],evaluation_used_for_training=False,
                reserved_final_seeds_unrun=state['reserved_final_seeds'],reproducibility=state['reproducibility'],
                validation_initial=state['validation_initial'])
    write_json(a.batch/'audit.json',report)
    print(json.dumps(dict(verified=True,updates=training['updates'],runs=len(results))))

if __name__=='__main__':main()
