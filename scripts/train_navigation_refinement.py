"""Fixed-budget BC comparison or on-policy local-goal PPO on audited real trials."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import torch
from minecraftrl.navigation_model import load_navigation,refinement_head,append_motion,value_head
from minecraftrl.navigation_refinement import validate_dataset,validate_bc_lineage,navigation_returns,correction_weights
from minecraftrl.rl_policy import guided_minibatch,cpu_tree,state_fingerprint
from minecraftrl.rl_opt import ppo_loss
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]


def load_run(path):
    summary=json.loads((path/'summary.json').read_text());audit=json.loads((path/'navigation-audit.json').read_text())
    probability=json.loads((path/'navigation-probability-audit.json').read_text());assert probability['verified']
    for name,sha in probability['files'].items():assert digest(path/name)==sha
    assert summary['execution_valid'] and summary['worker_exit_code']==0 and audit['verified']
    for name,sha in audit['files'].items():assert digest(path/name)==sha
    data=torch.load(path/'navigation-trajectory.pt',map_location='cpu')
    assert data['eligible'] and data['world']['experimental_split']==data['split']
    for r in data['rows']:
        assert torch.isfinite(r['features']).all() and not r['interrupted']
        assert len(r['motion_goal'])==10
    return data


def batch(rows):
    features=torch.cat([r['features'] for r in rows],1).cuda()
    goals=torch.tensor([r['motion_goal'] for r in rows],device='cuda').unsqueeze(0)
    actions={k:torch.cat([r['action'][k] for r in rows],1).cuda() for k in rows[0]['action']}
    return append_motion(features,goals),actions


def measure(head,rows,reference=None):
    values=[];kl=[];errors=[]
    with torch.no_grad():
        for start in range(0,len(rows),32):
            subset=rows[start:start+32];f,actions=batch(subset);pd=guided_minibatch(head,f)
            logp=head.logprob(actions,pd);values.extend((-logp).flatten().tolist())
            if reference is not None:
                old=guided_minibatch(reference,f);kl.extend(head.kl_divergence(old,pd).flatten().tolist())
            errors.extend(abs(v-r['old_logp']) for v,r in zip(logp.flatten().tolist(),subset))
    return dict(nll=sum(values)/len(values),labels=len(values),kl=sum(kl)/len(kl) if kl else None,
        stored_logp_max_error=max(errors))


def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['bc','bc_continue','ppo']);p.add_argument('output',type=Path)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--train',nargs='+',type=Path,required=True)
    p.add_argument('--development',nargs='*',type=Path,default=[])
    p.add_argument('--learning-rate',type=float);a=p.parse_args()
    if a.learning_rate is not None and (a.method!='bc_continue' or not 0<a.learning_rate<=.0001):
        p.error('learning-rate override only permits smaller positive bc_continue steps')
    out=a.output.resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('project output required')
    out.mkdir(parents=True,exist_ok=False);stop=ROOT/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,method=a.method,api_calls=0,game_actions=0,models=[])
    started=time.monotonic()
    try:
        torch.set_num_threads(1);torch.manual_seed(14101);torch.cuda.reset_peak_memory_stats()
        source=torch.load(a.actor,map_location='cpu');source_hash=digest(a.actor)
        parent=torch.load(ROOT/source['parent'],map_location='cpu')
        assert digest(ROOT/source['parent'])==source['parent_sha256']
        sys.path.insert(0,str(ROOT/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        from minecraftrl.policies import model_kwargs
        mapper=CameraHierarchicalMapping(n_camera_bins=11)
        original=make_action_head(DictType(**mapper.get_action_space_update()),2054,**model_kwargs(ROOT/'weights/steve1/2x.model')[1]).cuda().eval()
        original.load_state_dict(parent['actor']);original.requires_grad_(False)
        prior,refined=load_navigation(original,source);prior.eval().requires_grad_(False)
        train_data=[load_run(path) for path in a.train];dev_data=[load_run(path) for path in a.development]
        bc=a.method in ('bc','bc_continue')
        if a.method=='bc_continue':assert refined and 'navigation_value' in source
        for d in train_data:
            validate_dataset(d,'bc' if bc else 'ppo')
            if a.method=='bc_continue':
                behavior_path=(ROOT/d['config']['actor']).resolve()
                assert behavior_path.is_relative_to(ROOT/'results') and digest(behavior_path)==d['actor_sha256']
                validate_bc_lineage(d,torch.load(behavior_path,map_location='cpu'),source)
            else:assert d['actor_sha256']==source_hash
        assert all(d['split']=='development' for d in dev_data)
        assert {d['world']['seed'] for d in train_data}.isdisjoint({d['world']['seed'] for d in dev_data})
        owners={'teacher'} if bc else {'student'}
        train=[r for d in train_data for r in d['rows'] if r['owner'] in owners]
        dev=[r for d in dev_data for r in d['rows'] if r['owner']=='teacher']
        assert train and (not bc or dev)
        weights=torch.tensor(correction_weights([r['motion_goal'] for r in train]),device='cuda') if bc else None
        report.update(train_labels=len(train),development_labels=len(dev),source_sha256=source_hash,
            inputs={str(path):digest(path/'navigation-trajectory.pt') for path in a.train+a.development})
        advantages=[];returns=[]
        if a.method=='ppo':
            assert refined
            for d in train_data:
                adv,ret=navigation_returns(d['rows']);advantages+=adv;returns+=ret
            adv=torch.tensor(advantages,device='cuda');adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            ret=torch.tensor(returns,device='cuda')
        for kind in (('linear','mlp') if a.method=='bc' else (source['kind'],)):
            torch.manual_seed(14101)
            head=refinement_head(prior,kind) if a.method=='bc' else copy.deepcopy(prior)
            # load_navigation freezes the inference head; only the new residual/visual projection can update.
            from minecraftrl.navigation_model import RefinedLinear
            for m in head.modules():
                if isinstance(m,RefinedLinear):
                    m.residual.requires_grad_(True)
                    if hasattr(m,'visual_projection'):m.visual_projection.requires_grad_(True)
            value=value_head().cuda()
            if a.method in ('ppo','bc_continue'):value.load_state_dict(source['navigation_value'])
            else:torch.nn.init.zeros_(value[-1].weight);torch.nn.init.zeros_(value[-1].bias)
            anchor=copy.deepcopy(head).requires_grad_(False)
            initial_value=state_fingerprint(value.state_dict())
            params=[v for v in head.parameters() if v.requires_grad]
            if a.method=='ppo':params+=list(value.parameters())
            rate=a.learning_rate if a.learning_rate is not None else (.003 if a.method=='bc' else .0001)
            optimizer=torch.optim.AdamW(params,lr=rate,weight_decay=0,foreach=False)
            before=measure(head,train,anchor)
            if a.method=='ppo':assert before['stored_logp_max_error']<1e-5,before
            model=dict(kind=kind,before_train=before,before_development=measure(head,dev) if dev else None,
                updates=0,epochs=[],trainable_parameters=sum(v.numel() for v in params),
                learning_rate=optimizer.param_groups[0]['lr'],batch_size=32,kl_coefficient=.01,
                initial_actor_state_sha256=state_fingerprint(head.state_dict()))
            epochs=24 if a.method=='bc' else 8 if a.method=='bc_continue' else 4
            model['epoch_budget']=epochs
            for epoch in range(epochs):
                saved_head=cpu_tree(head.state_dict());saved_value=cpu_tree(value.state_dict())
                order=torch.randperm(len(train)).tolist();losses=[]
                for start in range(0,len(order),32):
                    if stop.exists():raise InterruptedError('STOP during navigation training')
                    indices=order[start:start+32];subset=[train[i] for i in indices];f,actions=batch(subset)
                    optimizer.zero_grad(set_to_none=True);pd=guided_minibatch(head,f)
                    with torch.no_grad():reference=guided_minibatch(anchor,f)
                    logp=head.logprob(actions,pd).flatten();kl=head.kl_divergence(reference,pd).mean()
                    if bc:loss=-(logp*weights[indices]).mean()+.01*kl
                    else:
                        old=torch.tensor([r['old_logp'] for r in subset],device='cuda')
                        prediction=value(f[0:1]).flatten()
                        loss=ppo_loss(logp,old,adv[indices],.1)+.5*(prediction-ret[indices]).square().mean()+.01*kl
                    if not torch.isfinite(loss):raise ValueError('nonfinite navigation loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    if stop.exists():raise InterruptedError('STOP before navigation optimizer')
                    optimizer.step();model['updates']+=1;losses.append(float(loss))
                measured=measure(head,train,anchor)
                entry=dict(epoch=epoch+1,mean_loss=sum(losses)/len(losses),train=measured)
                model['epochs'].append(entry);print(json.dumps(dict(kind=kind,**entry)),flush=True)
                if a.method in ('ppo','bc_continue') and measured['kl']>.03:
                    head.load_state_dict(saved_head);value.load_state_dict(saved_value)
                    model['kl_guard_rollback_epoch']=epoch+1;break
            dest=out/kind;dest.mkdir()
            if a.method=='bc_continue':assert initial_value==state_fingerprint(value.state_dict())
            artifact={k:source[k] for k in ('parent','parent_sha256','embedding','embedding_sha256','model_sha256')}
            artifact.update(schema='navigation_refinement_v1',kind=kind,actor=cpu_tree(head.state_dict()),
                navigation_value=cpu_tree(value.state_dict()),source_sha256=source_hash,training_method=a.method,
                root_actor_unchanged=True,motion_dim=10,coordinate_source='privileged_server',training_seed=14101)
            torch.save(artifact,dest/'checkpoint.pt')
            reloaded,_=load_navigation(original,torch.load(dest/'checkpoint.pt',map_location='cpu'))
            f,_=batch(train[:4]);pd=guided_minibatch(head,f);again=guided_minibatch(reloaded,f)
            reload_error=max((pd[k]-again[k]).abs().max().item() for k in pd);assert reload_error==0
            model.update(after_train=measure(head,train,anchor),after_development=measure(head,dev) if dev else None,
                checkpoint_sha256=digest(dest/'checkpoint.pt'),reload_error=reload_error,
                actor_state_sha256=state_fingerprint(head.state_dict()),value_state_sha256=state_fingerprint(value.state_dict()),
                value_unchanged=initial_value==state_fingerprint(value.state_dict()))
            model['actor_weights_changed']=model['actor_state_sha256']!=model['initial_actor_state_sha256']
            model['retained_updates']=model['updates']-((len(train)+31)//32 if 'kl_guard_rollback_epoch' in model else 0)
            write_json(dest/'training.json',model);report['models'].append(model)
            del head,anchor,optimizer,reloaded,value;torch.cuda.empty_cache()
        assert state_fingerprint(original.state_dict())==state_fingerprint(parent['actor'])
        report.update(completed=True,root_unchanged=True)
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write_json(out/'training.json',report);publish_completion(stop,'RL training ended; navigation refinement finished.')


if __name__=='__main__':main()
