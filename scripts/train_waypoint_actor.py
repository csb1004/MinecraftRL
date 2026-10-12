"""Fixed-budget BC of a waypoint-conditioned copy of the original STEVE-1 action head."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided_minibatch,cpu_tree,state_fingerprint
from minecraftrl.waypoint_actor import GoalLinear,navigation_head,append_goal,focus_goal_head
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('features',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--goal-focused-from',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project output required')
    out.mkdir(parents=True,exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,updates=0,api_calls=0,game_actions=0);started=time.monotonic()
    try:
        torch.set_num_threads(1);torch.manual_seed(9921);torch.cuda.reset_peak_memory_stats()
        receipt=json.loads((a.features.parent/'features.json').read_text())
        assert receipt['completed'] and digest(a.features)==receipt['features_sha256']
        data=torch.load(a.features,map_location='cpu');assert data['schema']=='waypoint_features_v1'
        parent_path=root/data['parent'];assert digest(parent_path)==data['parent_sha256']
        parent=torch.load(parent_path,map_location='cpu')
        sessions={s['id']:s for s in data['sessions']}
        train=[r for r in data['rows'] if r['split']=='train'];dev=[r for r in data['rows'] if r['split']=='development']
        assert train and dev and len(train)+len(dev)==len(data['rows'])
        assert {sessions[r['session']]['seed'] for r in train}.isdisjoint({sessions[r['session']]['seed'] for r in dev})
        for r in data['rows']:assert sessions[r['session']]['split']==r['split']
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        original=make_action_head(DictType(**mapper.get_action_space_update()),2054,**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda().eval()
        original.load_state_dict(parent['actor']);original.requires_grad_(False)
        actor=navigation_head(original).requires_grad_(True)
        goal_params=[m.goal.weight for m in actor.modules() if isinstance(m,GoalLinear)]
        goal_ids={id(v) for v in goal_params};visual_params=[v for v in actor.parameters() if id(v) not in goal_ids]
        def stopped():
            if stop.exists():raise InterruptedError('STOP during waypoint BC')
        def batch(rows):
            f=torch.cat([r['features'] for r in rows],1).cuda()
            goals=torch.tensor([r['goal'] for r in rows],device='cuda').unsqueeze(0)
            actions={k:torch.cat([r['action'][k] for r in rows],1).cuda() for k in rows[0]['action']}
            return f,goals,actions
        def measure(rows,head,conditioned=True):
            sums=dict(nll=0.,buttons_accuracy=0.,camera_accuracy=0.,goal_reversal_tv=0.)
            with torch.no_grad():
                for start in range(0,len(rows),32):
                    stopped();subset=rows[start:start+32];f,g,actions=batch(subset)
                    pd=guided_minibatch(head,append_goal(f,g) if conditioned else f)
                    sums['nll']+=-head.logprob(actions,pd).sum().item()
                    for k in ('buttons','camera'):sums[k+'_accuracy']+=(pd[k].argmax(-1)==actions[k]).sum().item()
                    if conditioned:
                        reverse=g.clone();reverse[...,:2]*=-1
                        counter=guided_minibatch(head,append_goal(f,reverse))
                        sums['goal_reversal_tv']+=sum((pd[k].exp()-counter[k].exp()).abs().sum().item()*.25 for k in pd)
            return {k:v/len(rows) for k,v in sums.items()}
        with torch.no_grad():
            f,g,actions=batch(train[:32]);ref=guided_minibatch(original,f);zero=guided_minibatch(actor,append_goal(f,g))
            report['zero_init_max_error']=max((ref[k]-zero[k]).abs().max().item() for k in ref)
            assert report['zero_init_max_error']<1e-5
        epochs=12;goal_lr=1e-3;scale=1.
        if a.goal_focused_from:
            previous=torch.load(a.goal_focused_from,map_location='cpu')
            assert previous['schema']=='waypoint_actor_v1' and previous['parent_sha256']==data['parent_sha256']
            assert previous['features_sha256']==digest(a.features)
            actor.load_state_dict(previous['actor']);scale=.1;focus_goal_head(actor,scale)
            epochs=24;goal_lr=1e-2;visual_params=[]
            report['starting_checkpoint_sha256']=digest(a.goal_focused_from)
        parameter_groups=[dict(params=goal_params,lr=goal_lr)]
        if visual_params:parameter_groups.insert(0,dict(params=visual_params,lr=5e-5))
        optimizer=torch.optim.AdamW(parameter_groups,weight_decay=0,foreach=False)
        report['trainable_parameters']=sum(v.numel() for v in actor.parameters() if v.requires_grad)
        before=dict(train=measure(train,original,False),development=measure(dev,original,False))
        report['initial_conditioned']=dict(train=measure(train,actor),development=measure(dev,actor))
        report['before']=before;write_json(out/'training.json',report)
        torch.manual_seed(9921)
        for epoch in range(epochs):
            order=torch.randperm(len(train)).tolist();losses=[]
            for start in range(0,len(order),32):
                stopped();subset=[train[i] for i in order[start:start+32]];f,g,actions=batch(subset)
                optimizer.zero_grad(set_to_none=True);pd=guided_minibatch(actor,append_goal(f,g))
                with torch.no_grad():reference=guided_minibatch(original,f)
                nll=-actor.logprob(actions,pd).mean();kl=actor.kl_divergence(reference,pd).mean();loss=nll+.01*kl
                if not torch.isfinite(loss):raise ValueError('nonfinite navigation loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([v for v in actor.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
                stopped();optimizer.step();report['updates']+=1;losses.append(float(nll))
            entry=dict(epoch=epoch+1,updates=report['updates'],mean_train_minibatch_nll=sum(losses)/len(losses))
            with (out/'updates.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
            print(json.dumps(entry),flush=True)
        after=dict(train=measure(train,actor),development=measure(dev,actor))
        artifact=dict(schema='waypoint_actor_v1',actor=cpu_tree(actor.state_dict()),parent=data['parent'],
            parent_sha256=data['parent_sha256'],features_sha256=digest(a.features),goal_dim=6,
            embedding=data['embedding'],embedding_sha256=data['embedding_sha256'],model_sha256=data['model_sha256'],
            coordinate_source='privileged_server',updates=report['updates'],root_actor_unchanged=True,
            condition='relative_goal_on_frozen_steve1_features',training_seed=9921,visual_prior_scale=scale,
            visual_head_frozen=bool(a.goal_focused_from))
        torch.save(artifact,out/'checkpoint.pt');torch.save(cpu_tree(optimizer.state_dict()),out/'bc-optimizer.pt')
        clone=navigation_head(original);clone.load_state_dict(torch.load(out/'checkpoint.pt',map_location='cpu')['actor'])
        with torch.no_grad():
            f,g,actions=batch(dev[:32]);features=append_goal(f,g)
            p1=guided_minibatch(actor,features);p2=guided_minibatch(clone,features)
            reload_error=max((p1[k]-p2[k]).abs().max().item() for k in p1)
            assert reload_error==0
        assert state_fingerprint(original.state_dict())==state_fingerprint(parent['actor'])
        assert digest(parent_path)==data['parent_sha256']
        report.update(completed=True,before=before,after=after,train_labels=len(train),development_labels=len(dev),
            epochs=epochs,visual_learning_rate=0. if a.goal_focused_from else 5e-5,goal_learning_rate=goal_lr,kl_coefficient=.01,
            visual_prior_scale=scale,visual_head_frozen=bool(a.goal_focused_from),
            parent_unchanged=True,reload_error=reload_error,checkpoint_sha256=digest(out/'checkpoint.pt'),
            features_sha256=digest(a.features),training_sessions=sorted({r['session'] for r in train}),
            development_sessions=sorted({r['session'] for r in dev}),
            actual_navigation_success=None,scope='Offline teacher action prediction; no game success claim')
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write_json(out/'training.json',report);publish_completion(stop,'RL training ended; waypoint actor BC finished.')
    print(json.dumps(report))


if __name__=='__main__':main()
