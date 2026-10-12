"""Summarize actual one-retry trials, separating initial success from recovery."""
import argparse
import json
from pathlib import Path
from minecraftrl.memory import atomic_json


def analyze(folder):
    folder = Path(folder)
    state = json.loads((folder / 'experiment.json').read_text(encoding='utf-8'))
    trials = []
    for row in state['reports']:
        report = row['report']
        events = [json.loads(line) for line in (folder / str(row['seed']) / 'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
        initial = next(e for e in events if e['event'] == 'command_acknowledged' and e.get('command_id') == 'begin-resume')
        retry = next((e for e in events if e['event'] == 'command_acknowledged' and e.get('command_id') == 'retry-once'), None)
        completed = next((e for e in events if e['event'] == 'inventory_target_reached'), None)
        observations = [e for e in events if e['event'] == 'inventory_observed']
        resets = [e for e in events if e['event'] == 'policy_reset_finished']
        trials.append(dict(
            seed=row['seed'], outcome=row['outcome'] if row.get('control_checks_passed') else 'technical_failure',
            retries=row['retries'], stalls=row['stalls'], steps=report['steps'],
            completion_after_initial_resume_s=None if completed is None else completed['monotonic_s']-initial['monotonic_s'],
            completion_after_retry_s=None if completed is None or retry is None else completed['monotonic_s']-retry['monotonic_s'],
            last_inventory={k:v for k,v in (observations[-1]['inventory'] if observations else {}).items() if v},
            progress=(report.get('task') or {}).get('progress'),
            worker_pid=report.get('worker_pid'), environment_resets=report.get('environment_resets'),
            policy_resets=len(resets), reset_conditions=[e['condition'] for e in resets],
            retry_task_preserved=row.get('retry_task_preserved'), condition_resampled=row.get('condition_resampled'),
            inference_ms=report.get('inference_ms'), control_cycle_ms=report.get('control_cycle_ms'),
            gpu=report.get('gpu'), device_vram=report.get('device_vram'),
            control_checks_passed=row.get('control_checks_passed', False)))
    result = dict(experiment_completed=state['completed'], source_unchanged=state.get('source_unchanged'),
                  planned=state['planned_runs'], reported=len(trials), unrun=state.get('unrun'),
                  api_calls=state['api_calls'], errors=state['errors'], trials=trials)
    result['counts'] = {name:sum(t['outcome']==name for t in trials) for name in (
        'initial_completed','recovered_after_retry','completed_on_resume_observation','not_completed','technical_failure')}
    result['valid_trials_with_retry'] = sum(t['retries'] > 0 and t['control_checks_passed'] for t in trials)
    result['interpretation'] = 'One saved world; prior embedding and recurrent state reset together. Initial successes do not establish recovery efficacy; no no-reset control arm.'
    atomic_json(folder / 'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    result = analyze(parser.parse_args().folder)
    print(json.dumps({k:v for k,v in result.items() if k != 'trials'}, indent=2))
