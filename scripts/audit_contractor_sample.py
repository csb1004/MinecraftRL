"""Decode one downloaded official sample on CPU; no model or game execution."""
import argparse,collections,json,sys
from pathlib import Path
import cv2
import numpy as np
from minecraftrl.contractor_actions import ContractorActions
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
manifest=json.loads((args.directory/'manifest.json').read_text(encoding='utf-8-sig'))
files={Path(f['file']).suffix:f for f in manifest['files']}
def local_file(item):return args.directory/item['url'].rsplit('/',1)[-1]
for item in files.values():assert digest(local_file(item))==item['sha256']
sys.path.insert(0,str(root/'external/steve1'))
from steve1.VPT.lib.actions import ActionTransformer
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
transformer=ActionTransformer(camera_binsize=2,camera_maxval=10,camera_mu=10,camera_quantization_scheme='mu_law')
mapper=CameraHierarchicalMapping(n_camera_bins=11)
video=cv2.VideoCapture(str(local_file(files['.mp4'])))
if not video.isOpened():raise ValueError('video unavailable')
fps=video.get(cv2.CAP_PROP_FPS);assert abs(fps-20)<.01
counts=collections.Counter();unknown=collections.Counter();start=None;last=None
converter=ContractorActions();shape=None
try:
    with local_file(files['.jsonl']).open(encoding='utf-8') as stream, (args.output/'aligned-actions.jsonl').open('x',encoding='utf-8') as target:
        for index,line in enumerate(stream):
            raw=json.loads(line);row=converter.convert(raw)
            ok,frame=video.read()
            if not ok:raise ValueError(f'missing video frame {index}')
            if shape is None:shape=list(frame.shape)
            if list(frame.shape)!=shape:raise ValueError('frame dimensions changed')
            start=raw if start is None else start;last=raw
            action=dict(row['action']);action['camera']=np.asarray(action['camera'],dtype=np.float32)
            factored=transformer.env2policy(action)
            mapped=mapper.from_factored({k:v[None] for k,v in factored.items()})
            assert all(np.isfinite(v).all() for v in mapped.values())
            roundtrip=mapper.to_factored(mapped)
            button_loss=not np.array_equal(roundtrip['buttons'][0],factored['buttons'])
            counts['button_projection_loss']+=button_loss
            row['bc_eligible']=row['bc_eligible'] and not button_loss
            row.update(frame_index=index,policy_action={k:v.tolist() for k,v in mapped.items()},
                       button_projection_loss=button_loss,goal_label=None,rm_context=None,training_ready=False)
            target.write(json.dumps(row)+'\n')
            counts['frames']+=1;counts['action_format_eligible']+=row['bc_eligible']
            counts['gui']+=row['gui'];counts['null']+=row['null_action']
            counts['attack_suppressed']+=row['initial_attack_suppression']
            counts['unsupported']+=bool(row['unsupported_actions']);unknown.update(row['unknown_keys'])
            counts['move']+=any(action[k] for k in ('forward','back','left','right'))
            counts['attack']+=action['attack']
            counts['fractional_camera']+=any(v!=int(v) for v in row['action']['camera'])
            if index in (0,1000,2000,3000,4000,5000):cv2.imwrite(str(args.output/f'frame-{index:05d}.jpg'),frame)
        # Preserve N+1 recordings rather than inventing an action for the final frame.
        extra_frames=0
        while video.read()[0]:extra_frames+=1
finally:video.release()
report=dict(completed=True,counts=dict(counts),unknown_keys=dict(unknown),fps=fps,shape=shape,
    duration_s=counts['frames']/fps,decoded_video_frames=counts['frames']+extra_frames,
    trailing_frames_without_action=extra_frames,pairing_status='offset_unverified',
    first_tick=start['tick'],last_tick=last['tick'],
    position_first=[start[k] for k in ('xpos','ypos','zpos')],position_last=[last[k] for k in ('xpos','ypos','zpos')],
    source_manifest_sha256=digest(args.directory/'manifest.json'),
    aligned_sha256=digest(args.output/'aligned-actions.jsonl'),
    converter_sha256=digest(root/'src/minecraftrl/contractor_actions.py'),
    upstream_mapper_sha256=digest(root/'external/steve1/steve1/VPT/lib/action_mapping.py'),
    optimizer_updates=0,policy_rollouts=0,training_ready=False,
    limitations=['All frames decoded and counts recorded; causal observation/action offset not independently calibrated.',
                'Goal and RM stage unknown; non-GUI action compatibility does not authorize training.',
                'Camera clipping/quantization and unsupported simultaneous buttons require loss audit.',
                'Initial attack correction is the upstream heuristic, not verified user intent.'])
write_json(args.output/'report.json',report);print(json.dumps(report))
