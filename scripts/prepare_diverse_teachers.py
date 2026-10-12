"""Single-use predeclared recovery demonstration fixtures, split before collection."""
import json
from pathlib import Path
from minecraftrl.rl_fixture import make_arena
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1];base=root/'results/diverse-teacher-worlds-20261008';base.mkdir(exist_ok=False)
sources=json.loads((root/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']
sources['104761']=dict(path=str(root/'results/rl-eval-104761-20261008T025450444322Z/run/initial-world.zip'))
cases=[]
for mirror,seed in ((False,104743),(True,104759)):
    for kind,end,start,yaw in [('far',5,(3.5,3.5),None),('corner',8,(6.3,1.3),180.),('long-corner',11,(3.5,1.3),180.)]:
        cases.append(dict(label=f'train-{seed}-{kind}',seed=seed,split='train',wall_end=end,start_xz=start,start_yaw=yaw,mirror_z=mirror))
cases.extend([dict(label='development-far',seed=104761,split='development',wall_end=6,start_xz=(4.5,3.5),start_yaw=None,mirror_z=False),
              dict(label='development-corner',seed=104761,split='development',wall_end=10,start_xz=(5.5,1.3),start_yaw=180.,mirror_z=True)])
write_json(base/'selection.json',dict(scope='Predeclared synthetic training/development cases, no evaluation trajectory reuse',cases=cases))
worlds={}
for c in cases:
    source=Path(sources[str(c['seed'])]['path']);path=base/(c['label']+'.zip')
    m=make_arena(source,path,'recovery',**{k:c[k] for k in ('wall_end','start_xz','start_yaw','mirror_z')})
    m.update(experimental_split=c['split'],curriculum_stage='diverse-correction',selection=c)
    write_json(path.with_suffix('.json'),m)
    worlds[c['label']]=dict(path=str(path),sha256=digest(path),seed=c['seed'],split=c['split'])
    write_json(base/'worlds.json',dict(completed=False,worlds=worlds));print(c['label'],flush=True)
write_json(base/'worlds.json',dict(completed=True,worlds=worlds))
