"""Read-only full-policy versus cached-prefix replay on one complete development episode."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail,TEMPORAL_SCOPE
from minecraftrl.rl_policy import RLPolicy,guided,state_fingerprint
from minecraftrl.temporal_bc import episode_rows,device_tree,recurrent_step,latent_features
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('features',type=Path)
    parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project-local output required')
    out.mkdir(exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    report=dict(completed=False,error=None,checkpoints=[],game_actions=0,optimizer_updates=0)
    try:
        torch.set_num_threads(1);torch.manual_seed(9603);torch.cuda.reset_peak_memory_stats()
        data=torch.load(args.features,map_location='cpu');groups=episode_rows(data)
        receipt=json.loads((args.features.parent/'features.json').read_text())
        assert receipt['completed'] and receipt['features_sha256']==digest(args.features)
        assert data['model_sha256']==digest(root/'weights/steve1/steve1.weights')
        session=next(s for s in data['plan']['sessions'] if s['split']=='development')
        rows=groups[session['id']];run=root/session['run']
        source=run/('correction-rows.json' if data['plan']['data_source']=='student_correction' else 'teacher-rows.json')
        assert digest(source)==session['rows_sha256'];original=json.loads(source.read_text())
        goal_path=root/data['plan']['embedding'];assert digest(goal_path)==data['goal_embedding_sha256']
        sys.path.insert(0,str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.agent import set_default_torch_device
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        space=DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
        kwargs,head_kwargs=model_kwargs(root/'weights/steve1/2x.model')
        policy=MinecraftAgentPolicy(space,kwargs,head_kwargs)
        weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu');policy.load_state_dict(weights);del weights
        policy=policy.cuda().eval().requires_grad_(False)
        from minecraftrl.cuda_startup import prime_convolution
        prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
        actor=make_action_head(space,rows[0]['features'].shape[-1],**head_kwargs).cuda().eval()
        agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=policy;agent.device=torch.device('cuda')
        f=RLPolicy.__new__(RLPolicy);f.agent=agent;f.torch=torch;f.device='cuda';f.reward_mode='wood-potential-v4'
        f.goal_embed=np.load(goal_path,allow_pickle=False)
        tail=torch.nn.Sequential(policy.net.lastlayer,policy.net.final_ln)
        for path in args.checkpoints:
            print('full-policy replay '+str(path),flush=True)
            training=json.loads((path.parent/'training.json').read_text())
            assert training['completed'] and training['checkpoint_sha256']==digest(path)
            if digest(path)!=data['parent_sha256']:
                assert training['features_sha256']==digest(args.features)
            saved=torch.load(path,map_location='cpu')
            assert saved['base_model_sha256']==data['model_sha256']
            load_tail(policy.net,dict(scope=TEMPORAL_SCOPE,backbone_tail=data['tail_initial'],backbone_temporal=data['temporal_initial']))
            load_tail(policy.net,saved);actor.load_state_dict(saved['actor'])
            # Upstream initial_state uses its process default device, not the module device.
            agent.hidden_state=device_tree(policy.initial_state(2),'cuda')
            memory=device_tree(data['temporal_initial_memory'],'cuda')
            max_error=memory_error=probability_error=0.
            for i,(row,source_row) in enumerate(zip(rows,original)):
                if stop.exists():raise InterruptedError('STOP during runtime replay')
                frame=run/source_row['frame'];assert digest(frame)==session['frame_sha256'][i]
                with Image.open(frame) as im:screen=np.asarray(im.convert('RGB'))
                f.first=i==0;f.set_reward_context(source_row['reward_context'])
                with torch.no_grad():
                    full,new_state=f.features(screen)
                    latent,memory=recurrent_step(policy.net.recurrent_layer.blocks[-1],row,memory,i)
                    cached=latent_features(tail,latent,row)
                    max_error=max(max_error,float((full-cached).abs().max()))
                    if state_fingerprint(new_state[-1])!=state_fingerprint(memory):memory_error+=1
                    a,b=guided(actor,full),guided(actor,cached)
                    probability_error=max(probability_error,max(float((a[k]-b[k]).abs().max()) for k in a))
                    agent.hidden_state=new_state
            assert max_error<1e-5 and memory_error==0 and probability_error<1e-5
            report['checkpoints'].append(dict(checkpoint=str(path),sha256=digest(path),frames=len(rows),session=session['id'],
                max_feature_error=max_error,memory_mismatches=memory_error,max_logprob_error=probability_error,scope=saved['scope']))
            write_json(out/'report.json',report)
        report['completed']=True
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        write_json(out/'report.json',report);publish_completion(stop,'RL training ended; temporal runtime replay finished; inspect report.')
    print(json.dumps(report));return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
