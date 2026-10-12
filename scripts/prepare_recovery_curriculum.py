"""Create separate curriculum archives from designated train/development worlds."""
import json
from pathlib import Path
from minecraftrl.rl_fixture import make_arena
from minecraftrl.world_store import digest
from minecraftrl.runtime import write_json

root=Path(__file__).resolve().parents[1];output=root/'results/recovery-curriculum-worlds-20261007'
output.mkdir(exist_ok=False)
old=json.loads((root/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']
items={}
for split,stage,end,seeds in [('train','short',3,[104743,104759]),('train','medium',5,[104743,104759]),('eval','long',7,[2000003,2000029])]:
    for i,seed in enumerate(seeds):
        source=Path(old[str(seed)]['path']);assert digest(source)==old[str(seed)]['sha256']
        label=f'{split}-{stage}-{seed}';path=output/(label+'.zip')
        result=make_arena(source,path,'recovery',wall_end=end,mirror_z=bool(i))
        result.update(experimental_split=split,curriculum_stage=stage)
        write_json(path.with_suffix('.json'),result)
        items[label]=dict(path=str(path),sha256=result['sha256'],seed=seed,split=split,stage=stage,mirror_z=bool(i))
        write_json(output/'worlds.json',dict(completed=False,worlds=items));print(label,flush=True)
write_json(output/'worlds.json',dict(completed=True,worlds=items))
