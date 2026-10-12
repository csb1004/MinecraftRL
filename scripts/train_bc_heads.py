"""Separate BC optimizer on reviewed, sequential features. Never reads eval rollouts."""
import argparse,copy,json,math,sys,traceback
from pathlib import Path
import torch
from minecraftrl.bc_core import validate_plan,comparison_gate
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import guided_minibatch as guided,cpu_tree
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('features',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tune-tail',action='store_true')
    p.add_argument('--learning-rate',type=float,default=1e-5)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if not math.isfinite(args.learning_rate) or not 0<args.learning_rate<=1e-5:raise ValueError('invalid bounded BC learning rate')
    if not out.is_relative_to(root/'results'):raise ValueError('project results only')
    out.mkdir(exist_ok=False,parents=True);stop=root/'results/STOP'
    report=dict(completed=False,updates=0,error=None,method='behavioral_cloning',runtime_comparison_eligible=False)
    claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        torch.set_num_threads(1);torch.manual_seed(9202);torch.cuda.reset_peak_memory_stats()
        data=torch.load(args.features,map_location='cpu')
        assert data['bc_schema']==1;validate_plan(data['plan'])
        freeze_rm=data['plan']['rm_context']=='unobserved_frozen'
        receipt=json.loads((args.features.parent/'features.json').read_text())
        assert receipt['completed'] and receipt['features_sha256']==digest(args.features)
        parent_path=root/data['plan']['parent'];assert digest(parent_path)==data['parent_sha256']
        parent=torch.load(parent_path,map_location='cpu');train=[];dev=[]
        splits={s['id']:s['split'] for s in data['plan']['sessions']}
        for row in data['rows']:
            assert row['split']==splits[row['session']]
            if freeze_rm:assert torch.count_nonzero(row['features'][...,-6:])==0
            assert torch.isfinite(row['features']).all()
            if args.tune_tail:
                assert row['tail_input'].shape==row['features'][...,:-6].shape
                assert torch.isfinite(row['tail_input']).all()
            (train if row['split']=='train' else dev).append(row)
        assert train and dev
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        dim=train[0]['features'].shape[-1]
        actor=make_action_head(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
                              dim,**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
        actor.load_state_dict(parent['actor']);actor.train()
        reference=copy.deepcopy(actor).eval().requires_grad_(False)
        tail=None
        if args.tune_tail:
            from types import SimpleNamespace
            from minecraftrl.policy_tail import load_tail,tail_state
            from steve1.embed_conditioned_policy import MinecraftAgentPolicy
            from steve1.VPT.lib.torch_util import set_default_torch_device
            set_default_torch_device('cpu')
            assert data['model_sha256']==digest(root/'weights/steve1/steve1.weights')
            skeleton=MinecraftAgentPolicy(DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update()),
                                           *model_kwargs(root/'weights/steve1/2x.model'))
            tail_net=SimpleNamespace(lastlayer=skeleton.net.lastlayer,final_ln=skeleton.net.final_ln)
            load_tail(tail_net,dict(scope='head_and_tail',backbone_tail=data['tail_initial']))
            tail=torch.nn.Sequential(tail_net.lastlayer,tail_net.final_ln).cuda().train()
            del skeleton
        parameters=list(actor.parameters())+([] if tail is None else list(tail.parameters()))
        optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0)
        def candidate_features(rows,base,active_tail):
            if active_tail is None:return base
            # Preserve runtime's per-step CFG shape and AMP precision.
            with torch.cuda.amp.autocast():
                latent=torch.cat([active_tail(r['tail_input'].cuda()) for r in rows],1)
            return torch.cat([latent.float(),base[...,-6:]],-1)
        def batch(rows):
            base=torch.cat([r['features'] for r in rows],1).cuda()
            a={k:torch.cat([r['action'][k] for r in rows],1).cuda() for k in rows[0]['action']}
            return candidate_features(rows,base,tail),a,base
        replay_error=0.
        if tail is not None:
            with torch.no_grad():
                for row in train+dev:
                    f,_,base=batch([row]);replay_error=max(replay_error,(f-base).abs().max().item())
            assert replay_error<1e-5,'cached tail does not reproduce original features'
        def measure(rows,head):
            total_nll=total_kl=0.
            with torch.no_grad():
                for i in range(0,len(rows),16):
                    if stop.exists():raise InterruptedError('STOP during BC measurement')
                    subset=rows[i:i+16];f,a,base=batch(subset);pd=guided(head,f)
                    total_nll+=-head.logprob(a,pd).sum().item()
                    total_kl+=head.kl_divergence(guided(reference,base),pd).sum().item()
            return dict(nll=total_nll/len(rows),reference_kl=total_kl/len(rows))
        before=dict(train=measure(train,actor),development=measure(dev,actor));history=[]
        torch.manual_seed(9202)  # Identical minibatch order for head-only and tail conditions.
        for epoch in range(2):
            order=torch.randperm(len(train)).tolist()
            for i in range(0,len(order),16):
                if stop.exists():raise InterruptedError('STOP before BC update')
                f,a,base=batch([train[j] for j in order[i:i+16]])
                optimizer.zero_grad(set_to_none=True);pd=guided(actor,f)
                with torch.no_grad():ref=guided(reference,base)
                nll=-actor.logprob(a,pd).mean();kl=actor.kl_divergence(ref,pd).mean();loss=nll+kl
                if not torch.isfinite(loss):raise ValueError('nonfinite BC loss')
                loss.backward()
                for module in actor.modules():
                    if freeze_rm and isinstance(module,torch.nn.Linear) and module.in_features==dim:
                        module.weight.grad[:,-6:]=0
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                if stop.exists():raise InterruptedError('STOP before BC optimizer step')
                optimizer.step();report['updates']+=1
                entry=dict(update=report['updates'],epoch=epoch,nll=nll.item(),reference_kl=kl.item())
                history.append(entry)
                with (out/'updates.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
        after=dict(train=measure(train,actor),development=measure(dev,actor));state=cpu_tree(actor.state_dict())
        delta=max((v-parent['actor'][k]).abs().max().item() for k,v in state.items())
        for k,v in state.items():
            if freeze_rm and v.ndim==2 and v.shape[-1]==dim:assert torch.equal(v[:,-6:],parent['actor'][k][:,-6:])
        candidate=dict(schema=1,scope='head_only',actor=state,value=parent['value'],
                       condition=parent['condition'],optimizer=cpu_tree(optimizer.state_dict()),
                       updates=parent.get('updates',0)+report['updates'],
                       bc=dict(method='behavioral_cloning',updates=report['updates'],goal=data['plan']['goal'],
                               features_sha256=digest(args.features),parent_sha256=data['parent_sha256'],
                               rm_columns_frozen=freeze_rm,value_frozen=True,ppo_optimizer_resumable=False))
        if tail is not None:
            candidate.update(scope='head_and_tail',backbone_tail=tail_state(tail_net),base_model_sha256=data['model_sha256'])
            report['tail_max_abs_change']=max((v-data['tail_initial'][name][k]).abs().max().item()
                for name,values in candidate['backbone_tail'].items() for k,v in values.items())
            assert report['tail_max_abs_change']>0
            report['tail_parameters']=sum(v.numel() for v in tail.parameters())
        torch.save(candidate,out/'checkpoint.pt')
        restored=torch.load(out/'checkpoint.pt',map_location='cpu');clone=copy.deepcopy(actor);clone.load_state_dict(restored['actor'])
        cloned_tail=None
        if tail is not None:
            cloned_net=copy.deepcopy(tail_net)
            # Start from the original tail so omission of trained weights cannot pass replay.
            load_tail(cloned_net,dict(scope='head_and_tail',backbone_tail=data['tail_initial']))
            load_tail(cloned_net,restored)
            cloned_tail=torch.nn.Sequential(cloned_net.lastlayer,cloned_net.final_ln)
        with torch.no_grad():
            f,a,base=batch(dev[:16]);restored_features=candidate_features(dev[:16],base,cloned_tail)
            reload_error=(actor.logprob(a,guided(actor,f))-clone.logprob(a,guided(clone,restored_features))).abs().max().item()
        assert reload_error==0
        assert all(torch.equal(v,restored['value'][k]) for k,v in parent['value'].items())
        report.update(completed=True,train_samples=len(train),development_samples=len(dev),before=before,after=after,
            runtime_comparison_eligible=comparison_gate(before['development']['nll'],after['development']['nll'],after['train']['reference_kl'],delta),
            scope=candidate['scope'],initial_tail_replay_error=replay_error,
            actor_max_abs_change=delta,reload_logp_error=reload_error,rm_columns_unchanged=all(torch.equal(v[:,-6:],parent['actor'][k][:,-6:]) for k,v in state.items() if v.ndim==2 and v.shape[-1]==dim),value_weights_unchanged=True,
            parent_sha256=data['parent_sha256'],checkpoint_sha256=digest(out/'checkpoint.pt'),
            features_sha256=digest(args.features),epochs=2,learning_rate=args.learning_rate,reference_kl_coefficient=1.,
            training_session_ids=sorted({r['session'] for r in train}),development_session_ids=sorted({r['session'] for r in dev}))
        assert set(report['training_session_ids']).isdisjoint(report['development_session_ids'])
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        publish_completion(stop,'RL training ended; BC pilot finished; inspect gate before rollout.')
        write_json(out/'training.json',report)
    print(json.dumps(report));return 0 if report['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
