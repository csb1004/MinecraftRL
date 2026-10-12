"""Locate first paired differences without replaying games or changing evidence."""
import json
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]; base = root/'results/goal-recovery-comparison-20261009'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
state = read(base/'experiment.json')


def transitions(run):
    with (run/'events.jsonl').open(encoding='utf-8') as source:
        for line in source:
            event = json.loads(line)
            if event['event']=='transition': yield event


for kind,comparison in state['comparisons'].items():
    target = base/(kind+'-prefix-differences.json')
    if target.exists(): continue
    runs = [Path(next(t['path'] for t in state['runs'] if t['kind']==kind and t['condition']==c))
            for c in ('wood','explore')]
    audits = [read(run/'goal-recovery-audit.json') for run in runs]
    limit = min(a['first_stall_step'] or a['steps'] for a in audits)
    result = dict(first_different_frame=None,first_different_action=None,first_different_server_pose=None,
        checked_prefix_steps=0,initial_state_identical=comparison['state_identical'],
        initial_pixel_identical=comparison['pixel_identical'],
        scope='Location of divergence, not attribution of its renderer/RNG cause')
    for step,(a,b) in enumerate(zip(*(transitions(run) for run in runs)),1):
        if step>limit: break
        result['checked_prefix_steps']=step
        if result['first_different_frame'] is None:
            frames=[run/'frames'/f'{step-1:06d}.png' for run in runs]
            if digest(frames[0]) != digest(frames[1]): result['first_different_frame']=step
        if result['first_different_action'] is None and a['action']!=b['action']:
            result['first_different_action']=step
        if result['first_different_server_pose'] is None:
            keys=('server_x','server_y','server_z','yaw','pitch')
            poses=[[event['info']['workshop'][k] for k in keys] for event in (a,b)]
            if poses[0]!=poses[1]: result['first_different_server_pose']=step
        if all(result[k] is not None for k in ('first_different_frame','first_different_action','first_different_server_pose')): break
    result['source_events_sha256']=[a['source_sha256']['events.jsonl'] for a in audits]
    result['implementation_sha256']=digest(__file__)
    write_json(target,result); print(json.dumps(dict(kind=kind,**result)))
