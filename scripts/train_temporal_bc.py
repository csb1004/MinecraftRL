"""Bounded paired BC pilot; fresh current-weight burn-in and eight-frame gradients."""
import argparse,copy,json,math,sys,time,traceback
from pathlib import Path
import torch
from minecraftrl.bc_core import comparison_gate
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail,tail_state,temporal_state,TEMPORAL_SCOPE
from minecraftrl.rl_policy import cpu_tree,guided,guided_minibatch
from minecraftrl.temporal_bc import episode_rows,windows,device_tree,recurrent_step,latent_features,forward_window
from minecraftrl.correction_bc import supervised_windows,supervised_batch
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('features',type=Path)
    parser.add_argument('--condition',choices=('tail','temporal'),required=True)
    parser.add_argument('--plan',type=Path,default=Path('configs/diverse-teacher-bc-20261008.json'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project-local output required')
    out.mkdir(exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,updates=0,runtime_comparison_eligible=False,condition=args.condition,
                learning_rate=5e-7,epochs=1,window_size=8,order_seed=9602,game_actions=0)
    started=time.monotonic()
    try:
        torch.set_num_threads(1);torch.manual_seed(9602);torch.cuda.reset_peak_memory_stats()
        data=torch.load(args.features,map_location='cpu');groups=episode_rows(data)
        receipt=json.loads((args.features.parent/'features.json').read_text())
        assert receipt['completed'] and receipt['features_sha256']==digest(args.features)
        assert data['model_sha256']==digest(root/'weights/steve1/steve1.weights')
        assert data['plan_sha256']==digest(root/args.plan)
        parent_path=root/data['plan']['parent'];assert digest(parent_path)==data['parent_sha256']
        parent=torch.load(parent_path,map_location='cpu')
        correction=data['plan']['data_source']=='student_correction'
        assert parent['scope']==(TEMPORAL_SCOPE if correction else 'head_only') and parent['condition']['reward_mode']=='wood-potential-v4'
        if correction:
            assert args.condition=='temporal','do not silently drop inherited temporal weights'
            assert data['parent_sha256']==data['plan']['checkpoint_sha256']
        assert data['goal_embedding_sha256']==digest(root/data['plan']['embedding'])
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        space=DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
        kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
        skeleton=MinecraftAgentPolicy(space,kwargs,head_kwargs)
        initial_checkpoint=dict(scope=TEMPORAL_SCOPE,backbone_tail=data['tail_initial'],backbone_temporal=data['temporal_initial'])
        load_tail(skeleton.net,initial_checkpoint)
        from types import SimpleNamespace
        block=skeleton.net.recurrent_layer.blocks[-1]
        block_count=len(skeleton.net.recurrent_layer.blocks)
        # Keep only the three trainable modules, not the frozen visual/prefix parameters.
        net=SimpleNamespace(lastlayer=skeleton.net.lastlayer,final_ln=skeleton.net.final_ln,
                            recurrent_layer=SimpleNamespace(blocks=[None]*(block_count-1)+[block]))
        tail=torch.nn.Sequential(net.lastlayer,net.final_ln)
        del skeleton
        block=block.cuda().eval().requires_grad_(args.condition=='temporal')
        tail=tail.cuda().eval().requires_grad_(True)
        actor=make_action_head(space,data['rows'][0]['features'].shape[-1],**head_kwargs).cuda().eval()
        actor.load_state_dict(parent['actor']);reference=copy.deepcopy(actor).eval().requires_grad_(False)
        params=list(tail.parameters())+list(actor.parameters())
        if args.condition=='temporal':params+=list(block.parameters())
        optimizer=torch.optim.AdamW(params,lr=5e-7,weight_decay=0,foreach=False)
        initial=data['temporal_initial_memory']
        sessions=data['plan']['sessions'];splits={s['id']:s['split'] for s in sessions}
        report.update(parent_sha256=data['parent_sha256'],features_sha256=digest(args.features),
                      model_sha256=data['model_sha256'],plan_sha256=data['plan_sha256'],
                      trainable_parameters=sum(p.numel() for p in params),
                      temporal_parameters=sum(p.numel() for p in block.parameters()),
                      masked_correction=correction,
                      label_counts={s:sum(r.get('loss_eligible',True) for k,rows in groups.items() if splits[k]==s for r in rows) for s in ('train','development')},
                      samples={s:sum(len(rows) for k,rows in groups.items() if splits[k]==s) for s in ('train','development')})
        def check_stop():
            if stop.exists():raise InterruptedError('STOP during temporal BC')
        def measure(compare_original=False,capture=False):
            result={s:dict(nll=0.,reference_kl=0.,count=0,all_frame_kl=0.,all_count=0,context_kl=0.,context_count=0) for s in ('train','development')}
            error=0.;captured=[]
            with torch.no_grad():
                for session,rows in groups.items():
                    memory=device_tree(initial,'cuda');totals=result[splits[session]]
                    for i,row in enumerate(rows):
                        check_stop();latent,memory=recurrent_step(block,row,memory,i)
                        f=latent_features(tail,latent,row);base=row['features'].cuda()
                        if compare_original:error=max(error,float((f-base).abs().max()))
                        pd=guided(actor,f);ref=guided(reference,base)
                        kl=actor.kl_divergence(ref,pd).item()
                        if not math.isfinite(kl):raise ValueError('nonfinite measurement')
                        totals['all_frame_kl']+=kl;totals['all_count']+=1
                        if row.get('loss_eligible',True):
                            action={k:v.cuda() for k,v in row['action'].items()}
                            nll=-actor.logprob(action,pd).item()
                            if not math.isfinite(nll):raise ValueError('nonfinite measurement')
                            totals['nll']+=nll;totals['reference_kl']+=kl;totals['count']+=1
                        else:totals['context_kl']+=kl;totals['context_count']+=1
                        if capture and splits[session]=='development':captured.append(cpu_tree(f))
            for totals in result.values():
                for key in ('nll','reference_kl'):totals[key]/=totals['count']
                totals['all_frame_kl']/=totals['all_count']
                if totals['context_count']:totals['context_kl']/=totals['context_count']
            return result,error,captured
        print('measure original sequential replay',flush=True)
        before,replay_error,_=measure(compare_original=True)
        report.update(before=before,initial_replay_max_error=replay_error)
        assert replay_error<1e-5,'original sequential replay mismatch'
        write_json(out/'training.json',report)
        if args.condition=='temporal':
            # Actual long-history window: measure temporal gradients without an optimizer update.
            probe=list(next(rows for session,rows in groups.items() if splits[session]=='train'))
            if correction:
                first_label=next(i for i,r in enumerate(probe) if r['loss_eligible'])
                start,end=first_label-7,first_label+1
            else:start,end=128,136
            for i in range(start,end):
                probe[i]=dict(probe[i],temporal_input=probe[i]['temporal_input'].cuda().detach().requires_grad_(True))
            with torch.no_grad():
                original=forward_window(block,tail,probe,initial,start,end)
                altered=list(probe);altered[end-1]=dict(probe[end-1],temporal_input=probe[end-1]['temporal_input']+1)
                future=forward_window(block,tail,altered,initial,start,end)
                report['future_to_past_max_error']=float((original[:,:-1]-future[:,:-1]).abs().max())
                assert report['future_to_past_max_error']==0
            del original,future,altered
            optimizer.zero_grad(set_to_none=True);features=forward_window(block,tail,probe,initial,start,end)
            action={k:v.cuda() for k,v in probe[end-1]['action'].items()}
            loss=-actor.logprob(action,guided(actor,features[:,-1:])).mean();loss.backward()
            report['earlier_frame_gradient_l1']=sum(float(probe[i]['temporal_input'].grad.abs().sum()) for i in range(start,end-1))
            report['temporal_gradient_l1']=sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None)
            assert report['earlier_frame_gradient_l1']>0 and report['temporal_gradient_l1']>0
            report['gradient_probe']=dict(start=start,end=end,context_frames=sum(not r.get('loss_eligible',True) for r in probe[start:end]),
                                          target_frames=1,burnin_frames=start)
            optimizer.zero_grad(set_to_none=True);del probe,features,action,loss
            print('long-history gradient/causality checks passed',flush=True)
            write_json(out/'training.json',report)
        chunks=[(session,start,end) for session,rows in groups.items() if splits[session]=='train'
                for start,end in supervised_windows(rows)]
        report['planned_updates']=len(chunks)
        torch.manual_seed(9602);order=torch.randperm(len(chunks)).tolist()
        for chunk_index in order:
            check_stop();session,start,end=chunks[chunk_index];rows=groups[session]
            optimizer.zero_grad(set_to_none=True)
            f=forward_window(block,tail,rows,initial,start,end)
            subset=rows[start:end];f,base,action=supervised_batch(subset,f)
            pd=guided_minibatch(actor,f)
            with torch.no_grad():ref=guided_minibatch(reference,base)
            nll=-actor.logprob(action,pd).mean();kl=actor.kl_divergence(ref,pd).mean();loss=nll+kl
            if not torch.isfinite(loss):raise ValueError('nonfinite BC loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            check_stop();optimizer.step();report['updates']+=1
            entry=dict(update=report['updates'],session=session,start=start,end=end,nll=nll.item(),
                       supervised_frames=sum(r.get('loss_eligible',True) for r in subset),
                       context_frames=sum(not r.get('loss_eligible',True) for r in subset),
                       reference_kl=kl.item(),gradient_norm=float(norm),elapsed_s=time.monotonic()-started)
            with (out/'updates.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
            if report['updates']%10==0:print(json.dumps(entry),flush=True)
            del f,base,action,pd,ref,nll,kl,loss
        optimizer.zero_grad(set_to_none=True)
        print('measure trained sequential policy',flush=True)
        after,_,captured=measure(capture=True)
        actor_state=cpu_tree(actor.state_dict());temporal=temporal_state(net);tail_saved=tail_state(net)
        actor_delta=max(float((v-parent['actor'][k]).abs().max()) for k,v in actor_state.items())
        temporal_delta=max(float((v-data['temporal_initial']['state'][k]).abs().max()) for k,v in temporal['state'].items())
        tail_delta=max(float((v-data['tail_initial'][name][k]).abs().max()) for name,state in tail_saved.items() for k,v in state.items())
        assert actor_delta>0 and tail_delta>0
        assert (temporal_delta>0) if args.condition=='temporal' else (temporal_delta==0)
        candidate=dict(schema=1,scope=TEMPORAL_SCOPE if args.condition=='temporal' else 'head_and_tail',
                       actor=actor_state,value=parent['value'],condition=parent['condition'],backbone_tail=tail_saved,
                       base_model_sha256=data['model_sha256'],updates=parent.get('updates',0)+report['updates'],
                       bc=dict(method='masked_truncated_sequence_bc' if correction else 'truncated_sequence_bc',updates=report['updates'],window=8,burnin='episode_start_current_weights',
                               features_sha256=report['features_sha256'],parent_sha256=data['parent_sha256'],
                               value_frozen=True,ppo_optimizer_resumable=False))
        if args.condition=='temporal':candidate['backbone_temporal']=temporal
        torch.save(candidate,out/'checkpoint.pt')
        torch.save(cpu_tree(optimizer.state_dict()),out/'bc-optimizer.pt')
        # Reset every updated module before reload, then compare complete development episodes.
        load_tail(net,initial_checkpoint);actor.load_state_dict(parent['actor'])
        restored=torch.load(out/'checkpoint.pt',map_location='cpu')
        load_tail(net,restored);actor.load_state_dict(restored['actor'])
        reloaded,_,second=measure(capture=True)
        reload_error=max(float((a-b).abs().max()) for a,b in zip(captured,second))
        assert len(captured)==len(second)==report['samples']['development'] and reload_error==0 and after==reloaded
        assert all(torch.equal(v,parent['value'][k]) for k,v in restored['value'].items())
        eligible=comparison_gate(before['development']['nll'],after['development']['nll'],after['train']['reference_kl'],actor_delta)
        report.update(completed=True,after=after,actor_max_abs_change=actor_delta,tail_max_abs_change=tail_delta,
                      temporal_max_abs_change=temporal_delta,reload_max_error=reload_error,
                      runtime_comparison_eligible=eligible,scope=candidate['scope'],checkpoint_sha256=digest(out/'checkpoint.pt'),
                      optimizer_sha256=digest(out/'bc-optimizer.pt'),
                      limitation='Eight-frame truncated gradients; cached frozen prefix; not full-episode BPTT or gameplay evidence')
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report.update(elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write_json(out/'training.json',report);publish_completion(stop,'RL training ended; temporal BC '+args.condition+' finished; inspect report.')
    print(json.dumps(report));return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
