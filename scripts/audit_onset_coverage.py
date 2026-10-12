"""Audit every new collection; admit complete teacher suffixes without moving dev data."""
import copy
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from minecraftrl.bc_core import validate_plan
from minecraftrl.rl_core import SHAPED_REWARDS
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import jsonable, write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/onset-coverage-collection-20261009'
out = base/'final-audit.json'
plan_path = root/'configs/onset-coverage-bc-20261009.json'
if out.exists() or plan_path.exists():
    raise FileExistsError('one-use admission exists')
state = json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and state['active'] is None
assert len(state['runs']) == 8
checkpoint = torch.load(state['checkpoint'], map_location='cpu', weights_only=False)
groups = ('actor', 'value', 'backbone_tail', 'backbone_temporal')
expected = {k: state_fingerprint(checkpoint[k]) for k in groups}
del checkpoint
old_path = root/'configs/masked-correction-bc-20261009.json'
old = json.loads(old_path.read_text())
assert digest(state['checkpoint']) == old['checkpoint_sha256']
plan = copy.deepcopy(old)
plan.update(scope='Original four train/two development episodes plus admitted new training starts',
            source_plans=[dict(path=old_path.relative_to(root).as_posix(), sha256=digest(old_path))],
            added_experiment=(base/'experiment.json').relative_to(root).as_posix(),
            added_experiment_sha256=digest(base/'experiment.json'))
results, totals = [], Counter()
for trial in state['runs']:
    run, report = Path(trial['path']), trial['report']
    assert trial['audited'] and trial['split'] == 'train' and trial['seed'] in (104743, 104759)
    audit = json.loads(Path(trial['audit']).read_text())
    for name, sha in audit['files_sha256'].items():
        assert digest(run/name) == sha
    data = torch.load(run/'student-prefix.pt', map_location='cpu', weights_only=False)
    assert not data['eligible'] and data['scope'] == 'correction_student_context_only' and data['split'] == 'train'
    for k in groups:
        assert state_fingerprint(data['initial_heads'][k]) == expected[k]
    assert state_fingerprint(data['initial_recurrent_state']) == data['condition']['initial_recurrent_sha256']
    assert data['condition']['sampling_seed'] == trial['sampling_seed']
    rows = json.loads((run/'correction-rows.json').read_text())
    assert len(data['rows']) == audit['owners'].get('student', 0)
    for i, row in enumerate(data['rows']):
        assert rows[i]['owner'] == 'student' and not rows[i]['loss_eligible']
        assert row['env_action'] == rows[i]['action'] and row['reward'] == rows[i]['reward']
        assert row['reward_context'] == rows[i]['reward_context'] and row['first'] == (i == 0)
        context = torch.tensor(row['reward_context'], dtype=row['features'].dtype).view(1, 1, 6).expand(2, 1, 6)
        assert torch.equal(row['features'][..., -6:], context)
    del data
    world = report['initial_world']
    source = state['worlds'][trial['label']]
    assert world['seed'] == trial['seed'] and world['experimental_split'] == 'train'
    assert world['sha256'] == source['sha256'] and digest(source['path']) == source['sha256']
    assert json.loads(Path(source['path']).with_suffix('.json').read_text()) == world
    assert digest(run/'goal-embed.npy') == old['embedding_sha256']
    ox, oz = (16*v for v in world['chunk'])
    initial, pose = rows[0]['server_before'], world['start_pose']
    target = [ox+pose['relative_position'][0], 201., oz+pose['relative_position'][2]]
    assert all(abs(initial['server_'+axis]-v) < 1e-4 for axis, v in zip('xyz', target))
    assert abs((initial['yaw']-pose['yaw']+180)%360-180) < 1e-4
    reward = SHAPED_REWARDS['wood-potential-v4'](initial)
    for i, row in enumerate(rows):
        assert reward.context() == row['reward_context']
        expected_reward = reward.advance(row['server_after'], terminal=bool(row['terminal'] and report['status'] == 'environment_done'))
        assert jsonable(expected_reward) == row['reward_components']
        if i < len(rows)-1:
            assert not row['terminal']
        assert digest(run/row['frame']) == audit['frames'][i]['sha256']
        with Image.open(run/row['frame']) as im:
            assert im.size == (640, 360)
            im.verify()
    assert reward.completed == report['server_target_reached']
    eligible = bool(report['demonstration_candidate'] and report['teacher_assisted_target_reached'] and report['expert_labels'])
    onset_pose = None
    if eligible:
        h, = report['handoffs']
        first = rows[h['step']]
        assert first['owner'] == 'teacher' and first['loss_eligible']
        server = first['server_before']
        z = server['server_z']-oz
        canonical_yaw = 180-server['yaw'] if world['curriculum']['mirror_z'] else server['yaw']
        onset_pose = dict(frame=h['step'], x=server['server_x']-ox,
                          canonical_z=16-z if world['curriculum']['mirror_z'] else z,
                          yaw=server['yaw'], pitch=server['pitch'], mirror_z=world['curriculum']['mirror_z'],
                          canonical_yaw=(canonical_yaw+180)%360-180,
                          frame_sha256=audit['frames'][h['step']]['sha256'], reward_context=first['reward_context'])
        session = dict(id=trial['label'], split='train', seed=trial['seed'], run=run.relative_to(root).as_posix(),
            context_start=0, start=0, end=len(rows), files_sha256=audit['files_sha256'], frames=audit['frames'],
            eligible_for_future_masked_bc=True, expert_labels=report['expert_labels'], handoffs=report['handoffs'],
            rows_sha256=audit['files_sha256']['correction-rows.json'], events_sha256=audit['files_sha256']['events.jsonl'],
            frame_sha256=[f['sha256'] for f in audit['frames']])
        plan['sessions'].append(session)
    totals.update(episodes=1, frames=len(rows), teacher_labels=report['expert_labels'], admitted_episodes=int(eligible),
                  student_successes=int(report['student_target_reached']), assisted_successes=int(report['teacher_assisted_target_reached']))
    results.append(dict(label=trial['label'], eligible=eligible, owners=audit['owners'], onset_pose=onset_pose,
        student_success=report['student_target_reached'], teacher_assisted_success=report['teacher_assisted_target_reached'],
        actual_log_delta=audit['actual_log_delta'], elapsed_control_s=report['elapsed_control_s'],
        inference_ms=audit['inference_ms'], control_cycle_ms=audit['control_cycle_ms'], device_vram=report['device_vram'],
        loaded_weights_verified=True, post_stop_dispatches=0, released_held_keys=0, remaining_owned_pids=[]))
for path, sha in state['frozen'].items():
    assert digest(path) == sha
assert plan['sessions'][:len(old['sessions'])] == old['sessions']
validate_plan(plan)
if totals['admitted_episodes']:
    write_json(plan_path, plan)
write_json(out, dict(verified=True, totals=dict(totals), results=results,
    plan=plan_path.relative_to(root).as_posix() if totals['admitted_episodes'] else None,
    plan_sha256=digest(plan_path) if totals['admitted_episodes'] else None,
    old_sessions_unchanged=True, sources_unchanged=True, optimizer_updates=0,
    scope='Audited training demonstrations; no autonomous-policy improvement claim'))
print(json.dumps(dict(totals)))
