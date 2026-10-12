"""Summarize recorded paired trials without treating unrun trials as failures."""
import argparse
import json
from pathlib import Path
from minecraftrl.memory import atomic_json


def analyze(folder):
    folder=Path(folder)
    state=json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
    result={'experiment_completed':state['completed'],'source_unchanged':state.get('source_unchanged'),
            'api_calls':state['api_calls'],'profiles':{},'trials':[],
            'interpretation':state['interpretation'],'errors':state['errors']}
    paired={seed:{} for seed in state['seeds']}
    for row in state['reports']:
        report=row['report']; profile=row['profile']; seed=row['seed']
        # Recorded as attempted even on technical errors; those are not policy failures.
        valid=row.get('control_checks_passed') is True
        outcome='technical_failure' if not valid else row['outcome']
        events=[json.loads(x) for x in (folder/(profile+'-'+str(seed))/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
        resumed=next((e['monotonic_s'] for e in events if e['event']=='command_acknowledged' and e['op']=='resume'),None)
        completed=next((e['monotonic_s'] for e in events if e['event']=='inventory_target_reached'),None)
        actions=[e for e in events if e['event']=='policy_dispatch_started']
        observations=[e for e in events if e['event']=='inventory_observed']
        tail=actions[-100:]
        def active(value): return any(value) if isinstance(value,list) else bool(value)
        last_inventory={k:v for k,v in (observations[-1]['inventory'] if observations else {}).items() if v}
        item=dict(profile=profile,seed=seed,outcome=outcome,steps=report['steps'],
                  progress=(report.get('task') or {}).get('progress'),last_inventory=last_inventory,
                  completion_after_resume_s=None if resumed is None or completed is None else completed-resumed,
                  inference_ms=report.get('inference_ms'),control_cycle_ms=report.get('control_cycle_ms'),
                  gpu=report.get('gpu'),device_vram=report.get('device_vram'),
                  condition=row.get('condition'),last_100_action_count=len(tail),
                  last_100_attack_count=sum(active(e['action']['attack']) for e in tail),
                  last_100_movement_count=sum(any(active(e['action'][key]) for key in ('forward','back','left','right','jump')) for e in tail))
        result['trials'].append(item)
        if valid: paired[seed][profile]=outcome=='completed'
    for profile in ('short','official'):
        trials=[r for r in result['trials'] if r['profile']==profile]
        technical=sum(r['outcome']=='technical_failure' for r in trials)
        result['profiles'][profile]=dict(planned=len(state['seeds']),attempted=len(trials),unrun=len(state['seeds'])-len(trials),
            technical_failures=technical,completed=sum(r['outcome']=='completed' for r in trials),
            observed_noncompletion=sum(r['outcome'] not in ('completed','technical_failure') for r in trials),
            valid_policy_trials=len(trials)-technical)
    result['paired']={str(seed):pair for seed,pair in paired.items()}
    atomic_json(folder/'analysis.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path)
    result=analyze(parser.parse_args().folder)
    print(json.dumps({k:result[k] for k in ('experiment_completed','source_unchanged','profiles','paired','errors')},indent=2))
