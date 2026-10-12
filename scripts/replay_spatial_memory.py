"""Replay immutable development evidence into spatial memory; no new game or API."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from minecraftrl.spatial_memory import SpatialMemory
from minecraftrl.rm_planner import validate_packet
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve();out=a.output.resolve()
    if not all(v.is_relative_to(root/'results') for v in (run,out)):raise ValueError('project results required')
    out.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    rows=read('goal-recovery-rows.json');packet=read('planner-input-01.json');binding=packet['runtime_binding']
    m=SpatialMemory(binding['world_id'],binding['episode_id'],source='privileged_server',dimension='overworld')
    def observe(s):m.observe([float(s['server_'+v]) for v in 'xyz'],int(s['server_tick']))
    observe(rows[0]['server_before']);m.set_landmark('episode_start')
    checkpoints={};enriched=None
    for index,row in enumerate(rows,1):
        observe(row['server_after'])
        if index in (802,825,1025):checkpoints[str(index)]=m.summary()
        if index==binding['sequence']:
            enriched=deepcopy(packet);enriched['mode']='archived_review';enriched['spatial_memory']=m.summary()
            validate_packet(enriched);write_json(out/'planner-input-spatial.json',enriched)
    assert enriched is not None
    m.save(out/'spatial-memory.json')
    restored=SpatialMemory.load(out/'spatial-memory.json',world_id=m.world_id,source=m.source,dimension=m.dimension)
    assert restored.nodes==m.nodes and restored.edges==m.edges and restored.landmarks==m.landmarks
    result=dict(verified=True,source_run=str(run),observed_transitions=len(rows),summary=m.summary(),
        checkpoints=checkpoints,archived_planner_packet_verified=True,exact_reload=True,
        actual_new_games=0,api_calls=0,optimizer_updates=0,
        scope='Observed topology and planner input replay; not successful navigation or live LLM planning',
        source_sha256={name:digest(run/name) for name in ('goal-recovery-rows.json','planner-input-01.json')})
    write_json(out/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('summary','checkpoints','source_sha256')}))


if __name__=='__main__':main()
