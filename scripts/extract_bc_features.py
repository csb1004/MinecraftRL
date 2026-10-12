"""Sequential frozen STEVE-1 features for separate BC; no game or PPO data."""
import argparse,copy,gc,json,sys,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from minecraftrl.bc_core import validate_plan,loss_eligible
from minecraftrl.contractor_actions import ContractorActions
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_policy import RLPolicy,cpu_tree
from minecraftrl.rl_control import claim_automatic_stop,publish_completion
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project results only')
    plan=json.loads(args.plan.read_text());validate_plan(plan)
    out.mkdir(exist_ok=False,parents=True);stop=root/'results/STOP'
    report=dict(completed=False,error=None,sessions=[],optimizer_updates=0,game_actions=0)
    claim_automatic_stop(stop,out/'STOP-before.txt')
    try:
        torch.set_num_threads(1);torch.manual_seed(9201);torch.cuda.manual_seed_all(9201)
        torch.cuda.reset_peak_memory_stats();sys.path.insert(0,str(root/'external/steve1'))
        from steve1.config import MINECLIP_CONFIG,PRIOR_INFO
        from steve1.mineclip_code.load_mineclip import load
        from steve1.data.text_alignment.vae import load_vae_model
        from steve1.utils.embed_utils import get_prior_embed
        data_dir=root/plan['data_directory']
        manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8-sig'))
        assert manifest['completed']
        for item in manifest['files']:
            assert digest(data_dir/Path(item['file']).name)==item['sha256'] if '\\' not in item['file'] else digest(data_dir/item['url'].rsplit('/',1)[-1])==item['sha256']
        alignment=json.loads((root/plan['alignment_report']).read_text())
        for row in alignment['rows']:
            assert row['fps']==20 and row['decoded_frames']==row['rows']+1
            scores=row['comparisons'];assert max(scores,key=lambda k:scores[k]['correlation'])=='0'
            assert scores['0']['correlation']>.8
        cfg=copy.deepcopy(MINECLIP_CONFIG);cfg['ckpt']['path']=str(root/'weights/steve1/attn.pth')
        clip=load(cfg,device='cuda').eval()
        prior=load_vae_model(dict(PRIOR_INFO,model_path=str(root/'weights/steve1/steve1_prior.pt')))
        goal=get_prior_embed(plan['goal'],clip,prior,'cuda')
        np.save(out/'goal-embed.npy',goal)
        del clip,prior;gc.collect();torch.cuda.empty_cache()
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
        weights=torch.load(root/'weights/steve1/steve1.weights',map_location='cpu')
        policy.load_state_dict(weights);del weights
        policy=policy.cuda().eval().requires_grad_(False)
        from minecraftrl.cuda_startup import prime_convolution
        report['backend_preparation']=prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
        agent=MineRLConditionalAgent.__new__(MineRLConditionalAgent);agent.policy=policy;agent.device=torch.device('cuda')
        feature_policy=RLPolicy.__new__(RLPolicy);feature_policy.agent=agent;feature_policy.torch=torch
        feature_policy.device='cuda';feature_policy.goal_embed=goal;feature_policy.reward_mode='wood-potential-v4'
        feature_policy.reward_context=[0.]*6
        stored=[]
        for session in plan['sessions']:
            print('extract '+session['id'],flush=True)
            source=data_dir/(session['id']+'.jsonl');rows=[json.loads(x) for x in source.open()]
            if session['end']>len(rows):raise ValueError('segment beyond source')
            converter=ContractorActions();agent.hidden_state=policy.initial_state(2);feature_policy.first=True
            v=cv2.VideoCapture(str(source.with_suffix('.mp4')));kept=0;processed=0;masked=0
            try:
                for index,raw in enumerate(rows[:session['end']]):
                    if stop.exists():raise InterruptedError('STOP during feature extraction')
                    ok,frame=v.read()
                    if not ok:raise ValueError('missing frame')
                    converted=converter.convert(raw)
                    if index<session['context_start']:continue
                    # No GUI/cursor frames are permitted in this bounded navigation context.
                    if raw['isGuiOpen'] or converted['unknown_keys']:raise ValueError('unsupported visual context')
                    screen=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    features,state=feature_policy.features(screen)
                    agent.hidden_state=state;feature_policy.first=False;processed+=1
                    if index<session['start']:continue
                    action=dict(converted['action']);action['camera']=np.asarray(action['camera'],dtype=np.float32)
                    factored=transformer.env2policy(action)
                    mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
                    back=mapper.to_factored(mapped)
                    button_loss=not np.array_equal(back['buttons'][0],factored['buttons'])
                    if not loss_eligible(converted,button_loss=button_loss):masked+=1;continue
                    stored.append(dict(session=session['id'],split=session['split'],frame=index,
                        features=cpu_tree(features),action={k:torch.from_numpy(v).unsqueeze(1) for k,v in mapped.items()}))
                    kept+=1
                if not kept:raise ValueError('no eligible demonstration actions')
            finally:v.release()
            report['sessions'].append(dict(session,kept=kept,masked=masked,context_frames_processed=processed,
                                           action_sha256=digest(source),video_sha256=digest(source.with_suffix('.mp4'))))
            write_json(out/'features.json',report)
        bundle=dict(bc_schema=1,plan=plan,rows=stored,plan_sha256=digest(args.plan),
                    parent_sha256=digest(root/plan['parent']),goal_embedding_sha256=digest(out/'goal-embed.npy'),
                    model_sha256=digest(root/'weights/steve1/steve1.weights'),source_manifest_sha256=digest(data_dir/'manifest.json'))
        torch.save(bundle,out/'features.pt')
        report.update(completed=True,samples=len(stored),features_sha256=digest(out/'features.pt'),
                      parent_sha256=bundle['parent_sha256'],plan_sha256=bundle['plan_sha256'])
    except BaseException as exc:
        report['error']=repr(exc);(out/'traceback.txt').write_text(traceback.format_exc())
    finally:
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        publish_completion(stop,'RL training ended; BC feature extraction finished; inspect report.')
        write_json(out/'features.json',report)
    print(json.dumps(report));return 0 if report['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
