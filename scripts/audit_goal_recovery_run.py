"""Read-only replay of actual goal-conditioned actions, RM, graph and cleanup."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from minecraftrl.policies import model_kwargs
from minecraftrl.rl_core import SHAPED_REWARDS, tick_delta
from minecraftrl.rl_policy import guided, state_fingerprint
from minecraftrl.runtime import jsonable, write_json
from minecraftrl.stall_detection import StallDetector
from minecraftrl.stall_replay import wood_sample, log_count
from minecraftrl.world_store import digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def assert_reward_evidence(parts, raw, feature, event):
    # JSON lists and the trajectory's Python tuples have identical semantics.
    assert jsonable(parts) == jsonable(raw) == jsonable(feature) == jsonable(event)


def audit(run, checkpoint):
    root = Path(__file__).resolve().parents[1]
    out = run/'goal-recovery-audit.json'
    if out.exists():
        raise FileExistsError(out)
    report = read(run/'summary.json')
    assert report['execution_valid'] and report['worker_exit_code'] == 0
    assert report['controller'] == 'steve1_graph_goal_conditioning'
    assert report['rl_split'] == 'eval' and not report['rl_training_eligible'] and report['rl_updates'] == 0
    assert report['expert_labels'] == report['direct_action_overrides'] == 0
    assert not report['process_cleanup']['remaining_live_pids']
    release = read(run/'input-release-state.json')
    assert report['release'] == 'neutral_command_sent' and release['verified'] and release['held_key_count'] == 0
    saved = torch.load(checkpoint, map_location='cpu')
    data = torch.load(run/'trajectory.pt', map_location='cpu')
    assert data['scope'] == 'goal_conditioning_diagnostic' and not data['eligible'] and data['split'] == 'eval'
    fingerprints = {k:state_fingerprint(saved[k]) for k in ('actor','value','backbone_tail','backbone_temporal')}
    assert fingerprints == data['initial_head_fingerprints'] == report['initial_head_fingerprints']
    assert fingerprints == read(run/'initial-weights.json')['fingerprints']
    assert state_fingerprint(data['initial_recurrent_state']) == data['condition']['initial_recurrent_sha256']
    raw, decisions, switches = (read(run/name) for name in
        ('goal-recovery-rows.json','stall-decisions.json','goal-switches.json'))
    assert len(raw) == len(data['rows']) == len(decisions) == report['steps'] > 0
    assert switches == report['switches'] and sum(s['mode']=='intervention' for s in switches) <= 1
    embeds = {mode:hashlib.sha256(np.load(run/name,allow_pickle=False).tobytes()).hexdigest()
              for mode,name in (('wood','goal-embed.npy'),('intervention','intervention-embed.npy'))}
    assert embeds == data['condition']['intervention']['embedding_sha256']
    # Recompute actual sampled probabilities and the official action conversion.
    sys.path.insert(0,str(root/'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
    from steve1.VPT.lib.action_head import make_action_head
    from steve1.VPT.lib.actions import ActionTransformer
    from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
    mapper = CameraHierarchicalMapping(n_camera_bins=11)
    transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
    actor = make_action_head(DictType(**mapper.get_action_space_update()),
        data['rows'][0]['features'].shape[-1], **model_kwargs(root/'weights/steve1/2x.model')[1]).cuda()
    actor.load_state_dict(saved['actor']); actor.requires_grad_(False)
    del saved
    initial = read(run/'initial-state.json')
    previous = start = initial['server']
    rm = SHAPED_REWARDS[data['reward_config']['name']](start)
    detector = StallDetector(); detector.update(wood_sample(start,{},str(run)))
    stopped = accepted = False
    n = inference_count = dispatch_count = 0
    mode = 'wood'; first_stall = None; intervention_start = None; expected_switch = None
    max_logp = max_yaw = max_displacement = distance = 0.
    prefix = hashlib.sha256()
    trigger_pose = None; first_crossing = first_ready = None
    stages = {}; sums = dict(task=0.,shaping=0.,time_cost=0.,failure=0.)
    world = initial['initial_world']; fixture = 'curriculum' in world
    intervention_rows = 0
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        e = json.loads(line); kind = e['event']
        if kind in ('inference_started','policy_dispatch_started'):
            assert not stopped and not accepted
            inference_count += kind == 'inference_started'
            dispatch_count += kind == 'policy_dispatch_started'
        if kind == 'stop_acknowledged': stopped = True
        if kind == 'target_reached': accepted = True
        if kind != 'transition': continue
        n += 1; row = raw[n-1]; feature = data['rows'][n-1]
        assert not stopped and e['step'] == n
        switch = next((s for s in switches if s['before_action']==n),None)
        if switch:
            assert expected_switch == (switch['mode'],switch['reason'])
            assert switch['committed'] and switch['after_step'] == n-1 and switch['server_before'] == previous
            pixels = np.asarray(Image.open(run/switch['frame']))
            assert list(pixels.shape) == switch['input_image_shape']
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == switch['input_image_sha256']
            mode = switch['mode']; expected_switch = None
            if mode == 'intervention': intervention_start = previous['server_tick']
        assert row['mode'] == mode and row['embedding_sha256'] == embeds[mode]
        assert row['owner'] == e['controller'] == 'steve1_'+mode and not row['loss_eligible']
        assert row['action'] == feature['env_action'] == e['action']
        assert row['server_before'] == previous and row['server_after'] == e['info']['workshop']
        assert feature['goal_conditioning'] == dict(mode=mode,embedding_sha256=embeds[mode])
        assert row['reward_context'] == feature['reward_context'] == rm.context()
        expected_context = torch.tensor(rm.context(),dtype=feature['features'].dtype)
        assert torch.equal(feature['features'][0,0,-6:],expected_context)
        assert torch.equal(feature['features'][1,0,-6:],expected_context)
        with torch.no_grad():
            action = {k:v.cuda() for k,v in feature['action'].items()}
            logp = actor.logprob(action,guided(actor,feature['features'].cuda())).item()
        max_logp = max(max_logp,abs(logp-feature['old_logp']))
        mapped = {k:v[:,0].numpy() for k,v in feature['action'].items()}
        restored = transformer.policy2env(mapper.to_factored(mapped))
        assert jsonable(restored) == e['action']
        server = row['server_after']; tick_delta(previous,server)
        parts = rm.advance(server,terminal=e['done'])
        assert_reward_evidence(parts,row['reward_components'],feature['reward_components'],e['info']['rl_reward'])
        assert parts['total'] == row['reward'] == feature['reward'] == e['reward']
        assert row['terminal'] == feature['terminal'] == bool(e['done'] or rm.completed)
        for k in sums: sums[k] += parts[k]
        sample = wood_sample(server,e['action'],str(run),previous=previous,terminal=row['terminal'])
        decision = detector.update(sample); decision['step'] = n
        assert jsonable(decision) == decisions[n-1]
        stages[sample.stage] = stages.get(sample.stage,0)+1
        if first_stall is None:
            prefix.update(json.dumps(e['action'],sort_keys=True).encode())
            prefix.update(digest(run/row['frame']).encode())
        if not row['terminal']:
            if mode == 'intervention':
                intervention_rows += 1
                if sample.stage in ('mine','collect'): expected_switch = ('wood','wood_interaction_ready')
                elif sample.tick-intervention_start >= 200: expected_switch = ('wood','tick_budget')
            elif first_stall is None and decision['event'] == 'stalled':
                first_stall = jsonable(decision); trigger_pose = server
                expected_switch = ('intervention','graph_stalled')
        elif mode == 'intervention': intervention_rows += 1
        distance += math.hypot(server['server_x']-previous['server_x'],server['server_z']-previous['server_z'])
        if intervention_start is not None:
            max_yaw = max(max_yaw,abs((server['yaw']-trigger_pose['yaw']+180)%360-180))
            max_displacement = max(max_displacement,math.hypot(server['server_x']-trigger_pose['server_x'],server['server_z']-trigger_pose['server_z']))
            if first_ready is None and sample.stage in ('mine','collect'): first_ready = n
        if fixture and first_crossing is None and server['server_x'] >= world['chunk'][0]*16+8.3:
            first_crossing = n
        previous = server
    assert stopped and n == dispatch_count == report['steps'] and n <= inference_count <= n+1
    assert max_logp <= 2e-4, max_logp
    assert first_stall == report['trigger'] and intervention_rows <= 200
    assert all(report['server_tick_deltas'][i] == 1 for i in range(n))
    assert len(report['server_tick_deltas']) == n and sums == report['reward_component_sums']
    assert report['intervention_committed'] == bool(intervention_rows)
    for switch in switches:
        assert switch['committed'] == (switch['before_action'] <= n)
        if not switch['committed']:
            assert switch['before_action'] == n+1 and expected_switch == (switch['mode'],switch['reason'])
    gained = log_count(previous)-log_count(start)
    assert bool(gained>0) == rm.completed == accepted == report['server_target_reached']
    result = dict(verified=True,steps=n,status=report['status'],server_success=accepted,log_gain=gained,
        intervention_steps=intervention_rows,first_stall_step=first_stall['step'] if first_stall else None,
        prefix_action_frame_sha256=prefix.hexdigest(),switches=switches,
        max_post_intervention_yaw_delta=max_yaw,max_post_intervention_displacement=max_displacement,
        expansion_eligible=bool(intervention_rows and max_yaw>=20 and max_displacement>=.5),
        first_wall_crossing=first_crossing,first_post_intervention_wood_interaction=first_ready,
        raw_stages=stages,horizontal_path_blocks=distance,action_head_logp_max_error=max_logp,
        checkpoint_sha256=digest(checkpoint),elapsed_control_s=report['elapsed_control_s'],
        inference_ms=report['inference_ms'],control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],
        device_vram=report['device_vram'],post_stop_dispatches=0,post_stop_inferences=0,released_held_keys=0,
        remaining_owned_pids=[],initial_weights_verified=True,one_server_tick_per_action=True,
        source_sha256={name:digest(run/name) for name in ('summary.json','trajectory.pt','events.jsonl',
            'goal-recovery-rows.json','goal-switches.json','stall-decisions.json','initial-state.json')},
        scope='Actual neural gameplay diagnostic; privileged graph trigger; no optimizer updates or final evaluation')
    write_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('switches','source_sha256')}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('checkpoint',type=Path)
    a = p.parse_args(); audit(a.run,a.checkpoint)
