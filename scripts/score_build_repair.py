"""Score building runs with the repair rule (design docs/specs/2026-11-15-build-repair-design.md §5).

Every run of the named arms is replayed with build_repair.repair_outcome on its batch fixture's task: a valid game is a
success when the server records leave the blueprint filled and no stray block (fixture strays and wrong placements, minus
the server destroys). The same rule scores the repair arms and the round 4 reference arm, whose game ended when the
blueprint was filled. Totals count valid games only.
Usage: score_build_repair.py --batch B --arms A... --out B/repair-score.json -> prints one line per arm.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.build_repair import repair_outcome  # noqa: E402

COUNTS = ('strays_left', 'dug', 'broken', 'wrong', 'unknown_placements', 'unknown_destroys')


def score_run(run_dir, task):
    rows = [json.loads(line) for line in (run_dir / 'telemetry.jsonl').read_text(encoding='utf-8').splitlines() if line]
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    outcome = repair_outcome(task, rows)
    valid = bool(summary.get('execution_valid'))
    return dict(run=run_dir.name, valid=valid, success=valid and outcome['done'], filled=outcome['filled'],
                **{k: outcome[k] for k in COUNTS})


def score(batch, arms):
    batch = Path(batch); out = {}
    for arm in arms:
        runs = []
        for summary in sorted((batch / arm).glob('c*/summary.json')):
            run = summary.parent; task_id = int(run.name.split('-')[0][1:])
            fixture = json.loads((batch / 'fixtures' / ('task-%d.json' % task_id)).read_text(encoding='utf-8'))
            runs.append(dict(score_run(run, fixture['build']), task=task_id))
        valid = [r for r in runs if r['valid']]
        out[arm] = dict(games=len(runs), valid=len(valid), success=sum(r['success'] for r in runs),
                        **{k: sum(r[k] for r in valid) for k in COUNTS}, runs=runs)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    result = score(args.batch, args.arms)
    args.out.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    for arm, s in result.items():
        print('%s repair success %d/%d valid %d strays left %d dug %d broken %d wrong %d' % (
            arm, s['success'], s['games'], s['valid'], s['strays_left'], s['dug'], s['broken'], s['wrong']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
