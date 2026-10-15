"""Independent real-state, action mapping, probability and per-goal reward replay."""
import argparse,json,sys,math
from pathlib import Path
import torch
from minecraftrl.navigation_model import load_navigation,NavigationSwitch,value_head,append_motion
from minecraftrl.navigation_refinement import TrialState,local_targets,motion_goal
from minecraftrl.rl_policy import guided
from minecraftrl.rl_core import SHAPED_REWARDS
from minecraftrl.runtime import jsonable,write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
    report=json.loads((run/'summary.json').read_text());assert report['execution_valid'] and report['worker_exit_code']==0
    data=torch.load(run/'navigation-trajectory.pt',map_location='cpu')
    rows=json.loads((run/'navigation-rows.json').read_text());initial=json.loads((run/'initial-state.json').read_text())['server']
    config=data['config'];actor=root/config['actor'];assert digest(actor)==data['actor_sha256']
    saved=torch.load(actor,map_location='cpu');parent=torch.load(root/saved['parent'],map_location='cpu')
    assert digest(root/saved['parent'])==saved['parent_sha256']
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    from minecraftrl.policies import model_kwargs
    mapper=CameraHierarchicalMapping(n_camera_bins=11);transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    original=make_action_head(DictType(**mapper.get_action_space_update()),2054,**model_kwargs(root/'weights/steve1/2x.model')[1]).cuda().eval()
    original.load_state_dict(parent['actor']);head,refined=load_navigation(original,saved)
    switch=NavigationSwitch(original,head.eval(),refined).requires_grad_(False)
    critic=None
    if 'navigation_value' in saved:
        critic=value_head().cuda().eval().requires_grad_(False);critic.load_state_dict(saved['navigation_value'])
    state=TrialState(local_targets(data['world'],config['points']),config['mode'],student_ticks=config.get('student_ticks',32));state.start(initial)
    root_rm=SHAPED_REWARDS['wood-potential-v4'](initial);previous=None;error=0.;reward_error=0.;value_error=0.
    assert len(rows)==len(data['rows'])==report['steps']
    with torch.no_grad():
        for r,t in zip(rows,data['rows']):
            assert not r['interrupted'] and r['owner']==t['owner']==state.owner
            assert r['index']==t['index']==state.index and r['target']==state.target
            g=motion_goal(r['server_before'],r['target'],previous,remaining=100-state.count)
            assert g==r['goal']==t['motion_goal'];switch.goal=g
            f=t['features'].cuda()
            expected=torch.tensor(root_rm.context(),device='cuda').view(1,1,6).expand(2,1,6)
            assert torch.equal(f[...,-6:],expected)
            pd=guided(switch,f);actions={k:v.cuda() for k,v in t['action'].items()}
            assert all(torch.isfinite(v).all() for v in pd.values())
            actual=switch.logprob(actions,pd).item()
            assert math.isfinite(actual) and math.isfinite(t['old_logp'])
            error=max(error,abs(actual-t['old_logp']))
            actual_value=critic(append_motion(f[0:1],g)).item() if critic is not None else 0.
            assert math.isfinite(actual_value) and math.isfinite(t['value'])
            value_error=max(value_error,abs(actual_value-t['value']))
            restored=transformer.policy2env(mapper.to_factored({k:v[:,0].numpy() for k,v in t['action'].items()}))
            assert jsonable(restored)==r['action']==t['env_action']
            result=state.advance(r['server_before'],r['server_after'])
            assert result==r['navigation_reward']==t['navigation_reward']
            root_reward=root_rm.advance(r['server_after'],terminal=r['root_terminal'])
            reward_error=max(reward_error,abs(root_reward['total']-r['root_reward']))
            previous=r['server_before']
    assert error<1e-5 and value_error<1e-5 and reward_error<1e-10 and state.receipts==data['receipts']==report['navigation_receipts']
    result=dict(verified=True,transitions=len(rows),stored_logp_max_error=error,root_reward_max_error=reward_error,
        actual_action_mapping_equal=True,per_goal_reward_equal=True,receipts_equal=True,
        stored_value_max_error=value_error,
        teacher_success_separate=True,files={n:digest(run/n) for n in ('navigation-trajectory.pt','navigation-rows.json','summary.json')})
    write_json(run/'navigation-probability-audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
