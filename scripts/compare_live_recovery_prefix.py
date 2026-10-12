"""Locate pre-intervention divergence from fixed replies without rerunning games."""
import json
from pathlib import Path

from PIL import Image

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def transitions(run):
    with (run/'events.jsonl').open(encoding='utf-8') as source:
        for line in source:
            event=json.loads(line)
            if event['event']=='transition':yield event


def main():
    root=Path(__file__).resolve().parents[1];base=root/'results/planning-recovery-live-20261009'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    state=read(base/'work.json');assert state['completed']
    live=Path(state['run']);output=base/'prefix-comparison.json'
    if output.exists():raise FileExistsError(output)
    results=[]
    for stamp in ('20261009T104906245333Z','20261009T110944517695Z'):
        fixed=root/f'results/goal-recovery-eval-104761-{stamp}/run';runs=[fixed,live]
        summaries=[read(p/'summary.json') for p in runs];initial=[read(p/'initial-state.json') for p in runs]
        keys=('server_x','server_y','server_z','yaw','pitch','server_inventory','world_time')
        conditions=('initial_recurrent_sha256','embedding_sha256','sampling_seed')
        row=dict(fixed=str(fixed),live=str(live),initial_server_equal=
            all(initial[0]['server'][k]==initial[1]['server'][k] for k in keys),
            initial_condition_equal=all(summaries[0]['condition'][k]==summaries[1]['condition'][k] for k in conditions),
            first_different_input_frame=None,first_different_action=None,first_different_pose=None,checked_steps=0)
        limit=min(s['trigger']['step'] if s['trigger'] else s['steps'] for s in summaries)
        for step,(a,b) in enumerate(zip(*(transitions(p) for p in runs)),1):
            if step>limit:break
            row['checked_steps']=step
            if row['first_different_input_frame'] is None:
                with Image.open(fixed/'frames'/f'{step-1:06d}.png') as x,Image.open(live/'frames'/f'{step-1:06d}.png') as y:
                    if x.size!=y.size or x.tobytes()!=y.tobytes():row['first_different_input_frame']=step
            if row['first_different_action'] is None and a['action']!=b['action']:row['first_different_action']=step
            if row['first_different_pose'] is None and any(a['info']['workshop'][k]!=b['info']['workshop'][k] for k in keys[:5]):
                row['first_different_pose']=step
            if all(row[k] is not None for k in ('first_different_input_frame','first_different_action','first_different_pose')):break
        row['source_events_sha256']=[digest(p/'events.jsonl') for p in runs];results.append(row)
    write_json(output,dict(comparisons=results,new_game_actions=0,new_api_calls=0,
        scope='Pre-intervention divergence locations; no renderer/RNG cause attribution or causal LLM effect',
        script_sha256=digest(Path(__file__))))
    print(json.dumps(results))


if __name__=='__main__':main()
