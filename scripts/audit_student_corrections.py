"""Freeze mixed-controller masks and verify actual loaded student weights once."""
import json
from collections import Counter
from pathlib import Path
import torch
from PIL import Image

from minecraftrl.rl_core import SHAPED_REWARDS
from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import jsonable, write_json
from minecraftrl.world_store import digest


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'results/student-correction-20261009'
    out = base/'final-audit.json'
    manifest_path = root/'configs/student-correction-data-20261009.json'
    if out.exists() or manifest_path.exists(): raise FileExistsError('one-use audit already exists')
    state = json.loads((base/'experiment.json').read_text())
    assert state['completed'] and not state['errors'] and state['active'] is None
    checkpoint = torch.load(state['checkpoint'], map_location='cpu', weights_only=False)
    groups = ('actor','value','backbone_tail','backbone_temporal')
    expected = {k:state_fingerprint(checkpoint[k]) for k in groups}
    del checkpoint
    totals, sessions, results = Counter(), [], []
    for trial in state['runs']:
        assert trial['audited']
        run = Path(trial['path']); report = trial['report']
        audit = json.loads(Path(trial['audit']).read_text())
        for name, sha in audit['files_sha256'].items(): assert digest(run/name) == sha
        data = torch.load(run/'student-prefix.pt', map_location='cpu', weights_only=False)
        assert not data['eligible'] and data['scope'] == 'correction_student_context_only'
        assert data['split'] == ('train' if trial['split']=='train' else 'eval')
        for k in groups: assert state_fingerprint(data['initial_heads'][k]) == expected[k]
        assert state_fingerprint(data['initial_recurrent_state']) == data['condition']['initial_recurrent_sha256']
        assert data['condition']['sampling_seed'] == 9701
        rows = json.loads((run/'correction-rows.json').read_text())
        assert len(data['rows']) == audit['owners'].get('student',0)
        for i, row in enumerate(data['rows']):
            assert rows[i]['owner']=='student' and not rows[i]['loss_eligible']
            assert row['env_action']==rows[i]['action'] and row['reward']==rows[i]['reward']
            assert row['reward_context']==rows[i]['reward_context'] and row['first']==(i==0)
            context=torch.tensor(row['reward_context'],dtype=row['features'].dtype).view(1,1,6).expand(2,1,6)
            assert torch.equal(row['features'][...,-6:],context)
        del data
        paths = Counter(); commands = {owner:Counter() for owner in ('student','teacher')}
        world = report['initial_world']; ox,oz = (x*16 for x in world['chunk'])
        assert world['seed']==trial['seed'] and world['experimental_split']==trial['split']
        assert world['sha256']==state['worlds'][trial['label']]['sha256']
        source_world=Path(state['worlds'][trial['label']]['path'])
        assert json.loads(source_world.with_suffix('.json').read_text())==world
        assert digest(run/'goal-embed.npy')=='810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40'
        initial=rows[0]['server_before'];pose=world['start_pose']
        expected_pose=[ox+pose['relative_position'][0],201.,oz+pose['relative_position'][2]]
        assert all(abs(initial['server_'+axis]-value)<1e-4 for axis,value in zip('xyz',expected_pose))
        assert abs((initial['yaw']-pose['yaw']+180)%360-180)<1e-4
        reward = SHAPED_REWARDS['wood-potential-v4'](rows[0]['server_before'])
        crossing = None
        for i,row in enumerate(rows):
            before, after = row['server_before'],row['server_after']
            assert reward.context()==row['reward_context']
            # Recovery evidence includes tuple coordinates; JSON stores them as lists.
            expected_reward=reward.advance(after,terminal=bool(row['terminal'] and report['status']=='environment_done'))
            assert jsonable(expected_reward)==row['reward_components']
            if i<len(rows)-1: assert not row['terminal']
            paths[row['owner']] += sum((after['server_'+a]-before['server_'+a])**2 for a in 'xz')**.5
            if crossing is None and after['server_x']>=ox+8.3:
                crossing = dict(step=i+1,owner=row['owner'],canonical_z=(16-(after['server_z']-oz)
                    if world['curriculum']['mirror_z'] else after['server_z']-oz))
            if row['owner'] in commands:
                for key in ('forward','back','left','right','jump','attack'):
                    value=row['action'].get(key,0)
                    commands[row['owner']][key]+=int(bool(value[0] if isinstance(value,list) else value))
            assert digest(run/row['frame'])==audit['frames'][i]['sha256']
            with Image.open(run/row['frame']) as im:
                assert im.size==(640,360)
                im.verify()
        assert reward.completed==report['server_target_reached']
        totals['episodes']+=1; totals['frames']+=len(rows)
        totals[trial['split']+'_frames']+=len(rows)
        totals[trial['split']+'_expert_labels']+=report['expert_labels']
        totals['student_successes']+=report['student_target_reached']
        totals['teacher_assisted_successes']+=report['teacher_assisted_target_reached']
        retreat = None
        if report['handoffs'] and crossing and crossing['owner']=='teacher':
            start = report['handoffs'][0]['step']
            before = rows[start]['server_before']['server_x']
            retreat = before-min(r['server_after']['server_x'] for r in rows[start:crossing['step']])
        sessions.append(dict(id=trial['label'],split=trial['split'],seed=trial['seed'],
            run=run.relative_to(root).as_posix(),context_start=0,start=0,end=len(rows),
            files_sha256=audit['files_sha256'],frames=audit['frames'],
            eligible_for_future_masked_bc=report['demonstration_candidate'],
            expert_labels=report['expert_labels'],handoffs=report['handoffs']))
        results.append(dict(label=trial['label'],owners=audit['owners'],
            student_success=report['student_target_reached'],teacher_assisted_success=report['teacher_assisted_target_reached'],
            actual_log_delta=audit['actual_log_delta'],handoffs=report['handoffs'],first_crossing=crossing,
            retreat_before_crossing_blocks=retreat,
            horizontal_path_blocks=dict(paths),commands={k:dict(v) for k,v in commands.items()},
            elapsed_control_s=report['elapsed_control_s'],inference_ms=audit['inference_ms'],
            control_cycle_ms=audit['control_cycle_ms'],device_vram=report['device_vram'],
            loaded_checkpoint_verified=True,post_stop_dispatches=0,released_held_keys=0,remaining_owned_pids=[]))
    for path,sha in state['frozen'].items(): assert digest(path)==sha
    for world in state['worlds'].values(): assert digest(world['path'])==world['sha256']
    checks=json.loads((base/'preflight-and-tests.json').read_text(encoding='utf-8-sig'))
    for path,sha in checks['source_sha256'].items(): assert digest(root/path)==sha
    assert {s['seed'] for s in sessions if s['split']=='train'}.isdisjoint(
        s['seed'] for s in sessions if s['split']=='development')
    write_json(manifest_path,dict(schema='student_correction_v1',data_source='student_correction',
        bc_training_ready=False,optimizer_updates=0,checkpoint_sha256=digest(state['checkpoint']),
        checkpoint=Path(state['checkpoint']).relative_to(root).as_posix(),goal='collect wood',reward_mode='wood-potential-v4',
        embedding='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        embedding_sha256='810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40',
        source_experiment=(base/'experiment.json').relative_to(root).as_posix(),
        source_experiment_sha256=digest(base/'experiment.json'),
        sequence_contract='Replay all frames from index 0; only loss_eligible teacher frames are targets',
        teacher_scope='privileged fixture geometry and server telemetry',sessions=sessions))
    write_json(out,dict(verified=True,totals=dict(totals),results=results,
        manifest=str(manifest_path),manifest_sha256=digest(manifest_path),source_hashes_preserved=True,
        optimizer_updates=0,scope='student-state corrective demonstrations; no trained-policy improvement claim'))
    print(json.dumps(dict(totals)))


if __name__=='__main__':main()
