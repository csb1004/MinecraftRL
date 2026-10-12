"""Compare old/new target interpretations on the same actual server snapshots."""
import argparse
import json
from collections import Counter
from pathlib import Path
from minecraftrl.rl_core import ShapedWoodRewardV2,ShapedWoodRewardV3,tick_delta
from minecraftrl.runtime import write_json


def audit(run):
    summary=json.loads((run/'summary.json').read_text())
    previous=json.loads((run/'initial-state.json').read_text())['server']
    initial_inventory=previous['server_inventory']
    count=0;active=0;old_wrong=0;old_stages=Counter();new_stages=Counter();changes=[];collection=[]
    mining_old_stages=Counter();mining_old_target_wrong=0;retention_checked=0
    stopped=accepted=False
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        event=json.loads(line);kind=event['event']
        if kind in ('inference_started','policy_dispatch_started'):assert not stopped and not accepted
        if kind=='stop_acknowledged':stopped=True
        if kind=='target_reached':accepted=True
        if kind!='transition':continue
        obs=event['info']['workshop'];tick_delta(previous,obs);count+=1
        old=ShapedWoodRewardV2.state(obs)[0];new=ShapedWoodRewardV3.state(obs)[0]
        old_stages[old]+=1;new_stages[new]+=1
        pos=lambda source,prefix:tuple(source[prefix+axis] for axis in 'xyz')
        target=pos(obs,'wood_v3_target_');prev_target=pos(previous,'wood_v3_target_')
        if obs['wood_server_mining']:
            active+=1
            mining_old_stages[old]+=1
            if pos(obs,'wood_target_')!=target:mining_old_target_wrong+=1
            assert obs['server_alive']==1 and obs['wood_v3_kind']==1
            assert target==pos(obs,'wood_mining_') and new=='mine'
            if old!='mine' or pos(obs,'wood_target_')!=target:old_wrong+=1
        elif previous['wood_v3_kind']==1 and obs['wood_v3_kind']==1:
            eye=tuple(obs['eye_'+axis] for axis in 'xyz')
            distance_squared=sum((a-b-.5)**2 for a,b in zip(eye,prev_target))
            if distance_squared<=32**2:
                assert target==prev_target
                retention_checked+=1
        if target!=prev_target or obs['wood_v3_kind']!=previous['wood_v3_kind']:
            changes.append(dict(step=count,old_kind=previous['wood_v3_kind'],kind=obs['wood_v3_kind'],
                                previous=prev_target,target=target,server_mining=obs['wood_server_mining']))
        # Losing a live target must keep the same pickup anchor, not the old locked block.
        if previous['wood_v3_kind']==1 and obs['wood_v3_kind']==2:
            assert target==prev_target
            collection.append(dict(step=count,target=target))
        previous=obs
    assert summary['execution_valid'] and summary['worker_exit_code']==0
    assert count==summary['steps'] and stopped
    assert not summary['process_cleanup']['remaining_live_pids']
    inventory_delta={k:v-initial_inventory.get(k,0) for k,v in previous['server_inventory'].items()
                     if v!=initial_inventory.get(k,0)}
    result=dict(verified=True,steps=count,status=summary['status'],success=summary['server_target_reached'],
                accepted_server_log_mining_snapshots=active,legacy_target_or_stage_mismatches=old_wrong,
                corrected_mining_mismatches=0,legacy_sensor_stages=dict(old_stages),new_sensor_stages=dict(new_stages),
                legacy_stages_during_server_mining=dict(mining_old_stages),
                legacy_target_mismatches_during_server_mining=mining_old_target_wrong,
                nonmining_live_target_retention_checks=retention_checked,
                target_changes=changes,broken_target_collection_transitions=collection,
                terminal_server={k:previous[k] for k in ('server_health','server_alive','held_key_count')},
                no_action_after_stop=True,scope='sensor comparison in same trajectory, not a learning effect')
    result['server_inventory_delta']=inventory_delta
    write_json(run/'wood-target-audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    audit(parser.parse_args().run)
