"""Eight new training-only arena copies; no policy or demonstration execution."""
import json
from pathlib import Path

from minecraftrl.rl_fixture import make_arena
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/onset-coverage-worlds-20261009'
base.mkdir(exist_ok=False)
source = json.loads((root/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']


def local(path):
    return root/str(path).replace('\\', '/').split('/MinecraftRL/', 1)[-1]


def windows(path):
    return 'D:/DATA/Desktop/MinecraftRL/'+path.relative_to(root).as_posix()


cases = []
for seed, mirror in ((104743, False), (104759, True)):
    for kind, pose, yaw, end in (('middle-oblique', (4.8,3.5), -110., 5),
                                 ('near-oblique', (5.9,4.8), -70., 8),
                                 ('edge-oblique', (6.3,1.3), 155., 11),
                                 ('inner-oblique', (5.8,2.0), -170., 8)):
        cases.append(dict(label=f'coverage-{seed}-{kind}', seed=seed, split='train',
                          start_xz=pose, start_yaw=yaw, wall_end=end, mirror_z=mirror,
                          sampling_seed=9702+len(cases)))
protocol = root/'docs/onset-coverage-protocol-2026-10-09.md'
write_json(base/'selection.json', dict(cases=cases, protocol_sha256=digest(protocol),
    scope='Training-only preparation, not rollout or acquisition evidence'))
worlds = {}
for case in cases:
    original = local(source[str(case['seed'])]['path'])
    assert digest(original) == source[str(case['seed'])]['sha256']
    archive = base/(case['label']+'.zip')
    manifest = make_arena(original, archive, 'recovery',
                          **{k:case[k] for k in ('wall_end','mirror_z','start_xz','start_yaw')})
    manifest.update(experimental_split='train', curriculum_stage='onset-coverage', selection=case)
    write_json(archive.with_suffix('.json'), manifest)
    worlds[case['label']] = dict(path=windows(archive), sha256=digest(archive), seed=case['seed'],
                                 split='train', sampling_seed=case['sampling_seed'])
    write_json(base/'worlds.json', dict(completed=False, worlds=worlds, game_actions=0))
    print(case['label'], flush=True)
write_json(base/'worlds.json', dict(completed=True, worlds=worlds, game_actions=0, actual_collection_runs=0))
