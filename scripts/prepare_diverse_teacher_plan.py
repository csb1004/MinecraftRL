"""Admit completed diverse teacher episodes, preserving source split and frame hashes."""
import collections,json
from pathlib import Path
from PIL import Image
from minecraftrl.bc_core import validate_plan
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/diverse-teacher-20261008'
output=root/'configs/diverse-teacher-bc-20261008.json'
if output.exists() or (base/'audit.json').exists():raise FileExistsError('completed admission exists')
batch=json.loads((base/'experiment.json').read_text());assert batch['completed'] and not batch['errors']
worlds=json.loads((root/'results/diverse-teacher-worlds-20261008/worlds.json').read_text())['worlds']
plan=json.loads((root/'configs/teacher-bc-20261008.json').read_text());plan['sessions']=[]
plan['scope']='Six train and two development synthetic teacher episodes; head/tail training not yet performed'
audits=[]
for r in batch['runs']:
    assert r['audited'];run=Path(r['path']);world=worlds[r['label']];summary=r['report'];w=summary['initial_world']
    assert digest(world['path'])==world['sha256']==w['sha256']
    assert r['split']==w['experimental_split'] and summary['demonstration_candidate']
    assert digest(run/'teacher-rows.json')==r['rows_sha256'] and digest(run/'events.jsonl')==r['events_sha256']
    assert not summary['process_cleanup']['remaining_live_pids'] and not summary['rl_training_eligible']
    rows=json.loads((run/'teacher-rows.json').read_text());assert len(rows)==summary['steps']
    g=rows[0]['server_before'];ox,oz=[v*16 for v in w['chunk']];pose=w['start_pose']
    expected=[ox+pose['relative_position'][0],201.,oz+pose['relative_position'][2]]
    assert all(abs(g['server_'+axis]-value)<1e-4 for axis,value in zip('xyz',expected))
    assert abs((g['yaw']-pose['yaw']+180)%360-180)<1e-4
    assert rows[-1]['terminal'] and not any(row['terminal'] for row in rows[:-1])
    from minecraftrl.rl_core import SHAPED_REWARDS
    reward=SHAPED_REWARDS['wood-potential-v4'](g);hashes=[]
    for row in rows:
        assert reward.context()==row['reward_context']
        assert reward.advance(row['server_after'],terminal=False)==row['reward_components']
        path=run/row['frame']
        with Image.open(path) as im:assert im.size==(640,360);im.verify()
        hashes.append(digest(path))
    assert reward.completed
    gain=rows[-1]['server_after']['server_inventory']['oak_log']-g['server_inventory']['oak_log'];assert gain>=1
    plan['sessions'].append(dict(id=run.parent.name,run=run.relative_to(root).as_posix(),split=r['split'],seed=r['seed'],
        context_start=0,start=0,end=len(rows),rows_sha256=digest(run/'teacher-rows.json'),
        events_sha256=digest(run/'events.jsonl'),frame_sha256=hashes))
    audits.append(dict(label=r['label'],split=r['split'],steps=len(rows),elapsed_s=summary['elapsed_control_s'],
                       oak_log_gain=gain,initial_pose_verified=True,rm_replay_verified=True,
                       source_world_sha256=world['sha256'],stages=dict(collections.Counter(x['reward_components']['stage_before'] for x in rows))))
assert len(audits)==8 and sum(r['split']=='train' for r in audits)==6
validate_plan(plan);write_json(output,plan)
write_json(base/'audit.json',dict(verified=True,runs=audits,plan_sha256=digest(output),
                                label_quantization_rechecked=False,learned_policy_successes=0,optimizer_updates=0))
print(json.dumps(audits))
