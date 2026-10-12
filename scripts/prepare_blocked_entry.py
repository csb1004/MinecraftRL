"""Single-use blocked-start train fixtures; no policy execution."""
import json
from pathlib import Path
from minecraftrl.rl_fixture import make_arena
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
base=root/'results/blocked-entry-worlds-20261008'
base.mkdir(exist_ok=False)
old=json.loads((root/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']
worlds={}
for i,seed in enumerate((104743,104759)):
    source=Path(old[str(seed)]['path'])
    assert digest(source)==old[str(seed)]['sha256']
    path=base/f'train-blocked-{seed}.zip'
    manifest=make_arena(source,path,'recovery',wall_end=3,mirror_z=bool(i),start_xz=(6.5,3.5))
    manifest.update(experimental_split='train',curriculum_stage='blocked-entry',
                    direct_path_blocked=True,geometry_evidence='constant z=3.5 (mirror 12.5), wall x=[7,8)')
    write_json(path.with_suffix('.json'),manifest)
    worlds[str(seed)]=dict(path=str(path),sha256=digest(path),seed=seed,sampling_seed=801+i,split='train')
    write_json(base/'worlds.json',dict(completed=False,worlds=worlds))
write_json(base/'worlds.json',dict(completed=True,worlds=worlds))
print(base)
