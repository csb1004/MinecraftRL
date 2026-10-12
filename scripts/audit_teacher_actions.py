"""Measure exact official student action round-trip loss on teacher records."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/steve1'))
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from steve1.VPT.lib.actions import ActionTransformer
from steve1.VPT.agent import ACTION_TRANSFORMER_KWARGS
mapper=CameraHierarchicalMapping(n_camera_bins=11);transformer=ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
reports=[]
for run in a.runs:
    rows=json.loads((run/'teacher-rows.json').read_text());errors=[];button_loss=0;nonfinite=0
    for row in rows:
        action={k:np.asarray(v) for k,v in row['action'].items()};original=action['camera']
        nonfinite+=int(not np.isfinite(original).all() or np.max(np.abs(original))>10)
        factored=transformer.env2policy(action)
        mapped=mapper.from_factored({k:v[None] for k,v in factored.items()});back=mapper.to_factored(mapped)
        button_loss+=int(not np.array_equal(back['buttons'][0],factored['buttons']))
        restored=transformer.policy2env({k:v[0] for k,v in back.items()})
        errors.extend(np.abs(restored['camera']-original).tolist())
    reports.append(dict(run=str(run),rows=len(rows),rows_sha256=digest(run/'teacher-rows.json'),
                        button_loss_rows=button_loss,invalid_camera_rows=nonfinite,
                        camera_error_mean=float(np.mean(errors)),camera_error_p95=float(np.quantile(errors,.95)),
                        camera_error_max=float(np.max(errors))))
write_json(a.output,dict(runs=reports,scope='Offline label conversion; not closed-loop quantized replay'))
print(json.dumps(reports))
