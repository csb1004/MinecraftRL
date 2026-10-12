"""Frozen recurrent features from complete, audited local teacher episodes."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from minecraftrl.bc_core import validate_plan
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import RLPolicy,cpu_tree
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-tail',action='store_true')
    p.add_argument('--cache-temporal',action='store_true')
    p.add_argument('--reuse-features',type=Path);a=p.parse_args()
    if a.cache_temporal:a.cache_tail=True
    if a.reuse_features and not a.cache_temporal:raise ValueError('feature reuse requires temporal cache mode')
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();plan=json.loads(a.plan.read_text())
    validate_plan(plan)
    correction=plan.get('data_source')=='student_correction'
    if plan.get('data_source') not in ('local_teacher','student_correction') or not out.is_relative_to(root/'results'):raise ValueError('demonstration source/project output required')
    out.mkdir(exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,sessions=[],optimizer_updates=0,game_actions=0)
    try:
        torch.set_num_threads(1);torch.manual_seed(9301);torch.cuda.reset_peak_memory_stats()
        embed=root/plan['embedding'];assert digest(embed)==plan['embedding_sha256']
        goal=np.load(embed,allow_pickle=False);assert np.isfinite(goal).all()
        np.save(out/'goal-embed.npy',goal);assert digest(out/'goal-embed.npy')==plan['embedding_sha256']
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.actions import ActionTransformer
        from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS,set_default_torch_device
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cuda');mapper=CameraHierarchicalMapping(n_camera_bins=11)
        transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        kwargs,head=model_kwargs(root/'weights/steve1/2x.model')
        policy=MinecraftAgentPolicy(DictType(**mapper.get_action_space_update()),kwargs,head)
        weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');policy.load_state_dict(weights);del weights
        if correction:
            from minecraftrl.policy_tail import load_tail,TEMPORAL_SCOPE
            parent_path=root/plan['parent'];assert digest(parent_path)==plan['checkpoint_sha256']
            parent=torch.load(parent_path,map_location='cpu')
            assert parent['scope']==TEMPORAL_SCOPE and parent['base_model_sha256']==digest(root/'weights/steve1/steve1.weights')
            load_tail(policy.net,parent);del parent
        policy=policy.cuda().eval().requires_grad_(False)
        tail_cache={}
        if a.cache_tail:
            policy.net.lastlayer.register_forward_pre_hook(lambda module,args:tail_cache.update(x=args[0].detach().clone()))
        if a.cache_temporal:
            policy.net.recurrent_layer.blocks[-1].register_forward_pre_hook(
                lambda module,args:tail_cache.update(temporal=args[0].detach().clone()))
        from minecraftrl.cuda_startup import prime_convolution
        report['backend_preparation']=prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
        agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=policy;agent.device=torch.device('cuda')
        f=RLPolicy.__new__(RLPolicy);f.agent=agent;f.torch=torch;f.device='cuda';f.goal_embed=goal;f.reward_mode='wood-potential-v4'
        stored=[];reused={};cached=None
        if a.reuse_features:
            from minecraftrl.bc_feature_reuse import checked_reuse
            from minecraftrl.policy_tail import tail_state,temporal_state
            from minecraftrl.rl_policy import state_fingerprint
            receipt=json.loads((a.reuse_features.parent/'features.json').read_text())
            assert receipt['completed'] and digest(a.reuse_features)==receipt['features_sha256']
            source_audit=json.loads((a.reuse_features.parent/'source-replay-audit.json').read_text())
            assert source_audit['verified'] and source_audit['features_sha256']==receipt['features_sha256']
            cached=torch.load(a.reuse_features,map_location='cpu')
            reused=checked_reuse(cached,plan,parent_sha256=digest(root/plan['parent']),
                model_sha256=digest(root/'weights/steve1/steve1.weights'),embedding_sha256=digest(embed))
            for key,current in dict(tail_initial=tail_state(policy.net),temporal_initial=temporal_state(policy.net),
                temporal_initial_memory=cpu_tree(policy.initial_state(2)[-1])).items():
                assert state_fingerprint(current)==state_fingerprint(cached[key]),key
            report['reused_features']=dict(path=str(a.reuse_features),sha256=receipt['features_sha256'],
                sessions=list(reused),frames=sum(len(v) for v in reused.values()))
        for session in plan['sessions']:
            print('extract '+session['id'],flush=True)
            run=root/session['run'];source=run/('correction-rows.json' if correction else 'teacher-rows.json');assert digest(source)==session['rows_sha256']
            assert digest(run/'events.jsonl')==session['events_sha256']
            summary=json.loads((run/'summary.json').read_text());world=summary['initial_world']
            assert summary['demonstration_candidate'] and not summary['rl_training_eligible']
            assert not summary['process_cleanup']['remaining_live_pids']
            assert world['experimental_split']==session['split'] and world['seed']==session['seed']
            rows=json.loads(source.read_text());assert len(rows)==session['end']==summary['steps']
            events=[json.loads(line) for line in (run/'events.jsonl').open()]
            transitions=[e for e in events if e['event']=='transition'];assert len(transitions)==len(rows)
            acknowledged=False
            for e in events:
                if e['event']=='stop_acknowledged':acknowledged=True
                if acknowledged:assert e['event']!='policy_dispatch_started'
            assert acknowledged
            if session['id'] in reused:
                for index,row in enumerate(rows):
                    if stop.exists():raise InterruptedError('STOP during cached source verification')
                    assert row['action']==transitions[index]['action']
                    assert digest(run/row['frame'])==session['frame_sha256'][index]
                stored.extend(reused[session['id']])
                prior=next(s for s in receipt['sessions'] if s['id']==session['id'])
                report['sessions'].append(dict(prior,reused=True))
                write_json(out/'features.json',report)
                print('reuse verified '+session['id'],flush=True)
                continue
            agent.hidden_state=policy.initial_state(2);f.first=True;errors=[];masked=0
            for index,(row,event) in enumerate(zip(rows,transitions)):
                if stop.exists():raise InterruptedError('STOP during extraction')
                assert row['action']==event['action'] and (not index or row['server_before']==rows[index-1]['server_after'])
                frame=run/row['frame'];assert digest(frame)==session['frame_sha256'][index]
                screen=np.asarray(Image.open(frame).convert('RGB'))
                f.set_reward_context(row['reward_context']);features,state=f.features(screen)
                expected=torch.tensor(row['reward_context'],device=features.device,dtype=features.dtype).view(1,1,6).expand(2,1,6)
                assert torch.equal(features[...,-6:],expected)
                assert torch.isfinite(features).all();agent.hidden_state=state;f.first=False
                eligible=True
                if correction:
                    from minecraftrl.correction_bc import correction_owner
                    eligible=session['frames'][index]['loss_eligible']
                    assert row['loss_eligible']==eligible and row['owner']==correction_owner(session,index)
                    assert event['controller']==row['owner']
                target=None
                if eligible:
                    action={k:np.asarray(v) for k,v in row['action'].items()}
                    assert np.isfinite(action['camera']).all() and np.max(np.abs(action['camera']))<=10
                    factored=transformer.env2policy(action);mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
                    back=mapper.to_factored(mapped);assert np.array_equal(back['buttons'][0],factored['buttons'])
                    restored=transformer.policy2env({k:v[0] for k,v in back.items()})
                    errors.extend(np.abs(restored['camera']-action['camera']).tolist())
                    target={k:torch.from_numpy(v).unsqueeze(1) for k,v in mapped.items()}
                else:masked+=1
                stored.append(dict(session=session['id'],split=session['split'],frame=index,features=cpu_tree(features),
                                   action=target))
                if correction:stored[-1].update(loss_eligible=eligible,owner=row['owner'])
                if a.cache_tail:stored[-1]['tail_input']=cpu_tree(tail_cache['x'])
                if a.cache_temporal:stored[-1]['temporal_input']=cpu_tree(tail_cache['temporal'])
            assert max(errors)<=2.5,'camera quantization admission bound'
            report['sessions'].append(dict(id=session['id'],split=session['split'],kept=len(rows),masked=masked,labels=len(rows)-masked,camera_max_error=max(errors)))
            write_json(out/'features.json',report)
        bundle=dict(bc_schema=2 if correction else 1,plan=plan,rows=stored,plan_sha256=digest(a.plan),parent_sha256=digest(root/plan['parent']),
                    goal_embedding_sha256=digest(embed),model_sha256=digest(root/'weights/steve1/steve1.weights'))
        if a.cache_tail:
            from minecraftrl.policy_tail import tail_state
            bundle['tail_initial']=tail_state(policy.net)
        if a.cache_temporal:
            from minecraftrl.policy_tail import temporal_state
            bundle['temporal_initial']=temporal_state(policy.net)
            bundle['temporal_initial_memory']=cpu_tree(policy.initial_state(2)[-1])
        torch.save(bundle,out/'features.pt')
        report.update(completed=True,samples=len(stored),features_sha256=digest(out/'features.pt'),parent_sha256=bundle['parent_sha256'],plan_sha256=bundle['plan_sha256'])
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        write_json(out/'features.json',report);publish_completion(stop,'RL training ended; teacher feature extraction finished.')
    print(json.dumps(report));return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
