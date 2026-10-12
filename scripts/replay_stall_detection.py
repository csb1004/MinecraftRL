"""Replay a fixed manifest once, writing all evidence to a new output directory."""
import argparse
import json
from pathlib import Path
import re

from minecraftrl.stall_detection import StallDetector
from minecraftrl.stall_replay import file_sha256, replay_run


def validate_manifest(config):
    ids = set()
    if set(config['excluded_final_seeds']) != {3000017,3000029,3000047,3000061,3000073}:
        raise ValueError('final evaluation exclusions must be preserved')
    for spec in config['runs']:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', spec['id']) or spec['id'] in ids:
            raise ValueError('invalid or duplicate output identifier')
        ids.add(spec['id'])
        if set(spec['sha256']) != {'events.jsonl','initial-state.json','summary.json'}:
            raise ValueError('all source hashes are required')
        if spec['split'] not in ('diagnostic_train','development') or spec['seed'] in config['excluded_final_seeds']:
            raise ValueError('invalid replay partition')
    if not ids:
        raise ValueError('empty replay manifest')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(config)
    output = args.output.resolve()
    if root not in output.parents or output == root:
        raise ValueError('output must be a new directory inside the project')
    output.mkdir(exist_ok=False)
    before_stop = file_sha256(root / 'results/STOP')
    sources = [root/'src/minecraftrl'/name for name in ('stall_detection.py','stall_replay.py')]
    report = dict(completed=False, runs=[], error=None, mode='offline_only',
                  manifest_sha256=file_sha256(args.manifest),
                  implementation_sha256={str(p.relative_to(root)):file_sha256(p) for p in sources},
                  windows=StallDetector.windows, goal_budget_ticks=StallDetector.goal_budget_ticks,
                  scope='Known development sources, not an unseen final evaluation; no policy training')
    try:
        seen = set()
        for spec in config['runs']:
            run = (root / spec['run']).resolve()
            if root not in run.parents or run in seen:
                raise ValueError('invalid or repeated source')
            seen.add(run)
            for name, sha in spec['sha256'].items():
                if name not in ('events.jsonl','initial-state.json','summary.json') or file_sha256(run/name) != sha:
                    raise ValueError('source hash mismatch: ' + name)
            summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
            if summary['seed'] != spec['seed'] or summary['seed'] in config['excluded_final_seeds']:
                raise ValueError('source seed/split violation')
            result = replay_run(run, output/(spec['id']+'.jsonl'))
            for name, sha in spec['sha256'].items():
                if file_sha256(run/name) != sha:
                    raise ValueError('source changed during replay')
            result.update(id=spec['id'], split=spec['split'], role=spec['role'], reference=spec['reference'])
            report['runs'].append(result)
            print(json.dumps(dict(id=spec['id'],steps=result['transitions'],alarms=len(result['alarms']),
                                  first_alarm=result['alarms'][0]['step'] if result['alarms'] else None)),flush=True)
        assert before_stop == file_sha256(root / 'results/STOP')
        report['stop_preserved'] = True
        report['completed'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
