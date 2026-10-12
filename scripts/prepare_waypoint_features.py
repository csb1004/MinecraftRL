"""Recompute current frozen parent features; retain only explicit navigation labels."""
import argparse,json,sys,time,traceback
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail,TEMPORAL_SCOPE
from minecraftrl.rl_policy import cpu_tree
from minecraftrl.temporal_bc import episode_rows,device_tree,recurrent_step,latent_features
from minecraftrl.waypoints import label_waypoints
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project output required')
    out.mkdir(parents=True,exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,optimizer_updates=0,game_actions=0,sessions=[]);started=time.monotonic()
    try:
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        features=root/'results/onset-coverage-features-20261009/features.pt'
        parent_path=root/'results/onset-coverage-training-20261009/checkpoint.pt'
        assert digest(features)=='152953150fb427bd10a8ad924e76ad9a368e1bf794771b30f8d878243cd87270'
        assert digest(parent_path)=='473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
        data=torch.load(features,map_location='cpu');groups=episode_rows(data)
        parent=torch.load(parent_path,map_location='cpu')
        # Only the last recurrent block/tail/head changed; earlier cached inputs remain valid.
        assert parent['scope']==TEMPORAL_SCOPE and parent['base_model_sha256']==data['model_sha256']
        assert digest(root/'weights/steve1/steve1.weights')==data['model_sha256']
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu');mapper=CameraHierarchicalMapping(n_camera_bins=11)
        skeleton=MinecraftAgentPolicy(DictType(**mapper.get_action_space_update()),*model_kwargs(root/'weights/steve1/2x.model'))
        load_tail(skeleton.net,parent)
        block=skeleton.net.recurrent_layer.blocks[-1].cuda().eval().requires_grad_(False)
        tail=torch.nn.Sequential(skeleton.net.lastlayer,skeleton.net.final_ln).cuda().eval().requires_grad_(False)
        del skeleton,parent
        stored=[];source_hashes={}
        for session in data['plan']['sessions']:
            name=session['id'];print('replay '+name,flush=True)
            run=root/session['run'];source=run/'correction-rows.json'
            assert digest(source)==session['rows_sha256'];source_hashes[source.relative_to(root).as_posix()]=digest(source)
            source_rows=json.loads(source.read_text());summary=json.loads((run/'summary.json').read_text())
            world=summary['initial_world'];assert world['experimental_split']==session['split'] and world['seed']==session['seed']
            labels=label_waypoints(session,world,source_rows);memory=device_tree(data['temporal_initial_memory'],'cuda')
            with torch.no_grad():
                for i,row in enumerate(groups[name]):
                    if stop.exists():raise InterruptedError('STOP during waypoint feature replay')
                    latent,memory=recurrent_step(block,row,memory,i);f=latent_features(tail,latent,row)
                    if i in labels:
                        assert row['loss_eligible'] and row['action'] is not None
                        stored.append(dict(session=name,split=session['split'],frame=i,
                            features=cpu_tree(f),action=row['action'],**labels[i]))
            report['sessions'].append(dict(id=name,split=session['split'],seed=session['seed'],
                frames=len(groups[name]),navigation_labels=len(labels),run=session['run']))
            write_json(out/'features.json',report)
        artifact=dict(schema='waypoint_features_v1',rows=stored,sessions=report['sessions'],
            parent=str(parent_path.relative_to(root)),parent_sha256=digest(parent_path),
            source_features_sha256=digest(features),model_sha256=data['model_sha256'],
            embedding=data['plan']['embedding'],embedding_sha256=data['goal_embedding_sha256'],
            source_sha256=source_hashes,goal_source='privileged teacher waypoint and current server pose',
            coordinate_condition='privileged_server',supervision='teacher navigation only')
        torch.save(artifact,out/'features.pt')
        report.update(completed=True,features_sha256=digest(out/'features.pt'),parent_sha256=digest(parent_path),
            total_labels=len(stored),train_labels=sum(r['split']=='train' for r in stored),
            development_labels=sum(r['split']=='development' for r in stored),source_sha256=source_hashes)
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write_json(out/'features.json',report);publish_completion(stop,'RL training ended; waypoint feature preparation finished.')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sessions','source_sha256')}))


if __name__=='__main__':main()
