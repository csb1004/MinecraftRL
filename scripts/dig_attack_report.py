"""Dig-stage attack report of sensor repair runs (diagnosis 2026-11-24). Replays each run's sensor log through
SensorRepairFeed (as build2_data does) and, on the decision rows of the dig stage, counts where the client crosshair was
(the target cell, another solid, air) and how often attack was pressed there; per target, the longest unbroken attack run
on it against the rows the block needs (beacon 90, iron and diamond block 19, gold block 12; measured 2026-11-23).
Decision rows are the head's ('policy') and the teacher's ('scripted_probe'). Report only: no success claim.
Run in the WSL steve1 env: dig_attack_report.py --root R --batch results/repair2-20261123 dev/e-final beacon/s-final+e-final
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

NEED = {'beacon': 90, 'iron_block': 19, 'diamond_block': 19, 'gold_block': 12}
DECISIONS = ('policy', 'scripted_probe')


def run_report(run, fixture):
    from minecraftrl.build2_data import replay_info
    from minecraftrl.build_sensor_repair import DIG_PREFIX, SensorRepairFeed
    from minecraftrl.pipeline.seen_map import SeenMap
    from minecraftrl.pipeline.sensor import load_sensor_log, sensor_frame
    feed = SensorRepairFeed(fixture); seen = SeenMap(); sensor = list(load_sensor_log(run / 'sensor'))
    c = Counter(); runs = defaultdict(list); cur = None; kind_of = {}
    for line in open(run / 'telemetry.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r.get('controller') in DECISIONS:
            view = feed.decision_view(); attack = bool((r.get('action') or {}).get('attack'))
            stage = None if view is None else view.stage
            if stage is not None and feed.combined: stage = stage[len(DIG_PREFIX):] if stage.startswith(DIG_PREFIX) else None
            holding = None
            if stage == 'dig':
                f = feed.last_frame
                kind, cell = f.crosshair if f is not None else ('air', None)
                tgt = tuple(view.target)
                where = 'air' if kind == 'air' else 'target' if tuple(cell) == tgt else 'other'
                c['dig'] += 1; c[where] += 1; c[where + '_attack'] += attack
                if tgt in feed.dig.targets: kind_of[tgt] = feed.dig.targets[tgt][1]
                if where == 'target' and attack: holding = tgt
                elif where == 'target': c['released_on_target'] += 1
            if holding is not None and cur is not None and cur[0] == holding: cur[1] += 1
            else:
                if cur is not None: runs[cur[0]].append(cur[1])
                cur = None if holding is None else [holding, 1]
        if r.get('sensor_row') is not None:
            f = sensor_frame(replay_info(r, sensor[r['sensor_row']]), r['t'])
            if f is not None: seen.update(f); feed.observe_frame(f, seen)
        feed.observe(r)
    if cur is not None: runs[cur[0]].append(cur[1])
    targets = [dict(block=kind_of.get(t), need=NEED.get(kind_of.get(t)), longest=max(rl), bursts=len(rl))
               for t, rl in runs.items()]
    return dict(run=run.name, **{k: c[k] for k in ('dig', 'target', 'target_attack', 'other', 'other_attack', 'air',
                                                    'air_attack', 'released_on_target')}, targets=targets)


def fixture_of(run, batch, arm):
    request = json.loads((run / 'request.json').read_text(encoding='utf-8'))
    if 'fixture' in request: return json.loads(Path(request['fixture']).read_text(encoding='utf-8'))
    task = int(run.name.split('-')[0][1:])
    return json.loads((batch / arm.split('/')[0] / 'fixtures' / ('task-%d.json' % task)).read_text(encoding='utf-8'))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--batch', type=Path, required=True, help='batch root, e.g. results/repair2-20261123')
    p.add_argument('arms', nargs='+', help='<batch>/<arm> under the batch root, e.g. dev/e-final')
    p.add_argument('--json', type=Path, help='also write the report here')
    args = p.parse_args(argv)
    sys.path.insert(0, str(args.root / 'src'))
    batch = args.batch if args.batch.is_absolute() else args.root / args.batch
    report = {}
    for arm in args.arms:
        rows = [run_report(run, fixture_of(run, batch, arm)) for run in sorted((batch / arm).glob('c*'))
                if (run / 'summary.json').exists()]
        total = Counter()
        for r in rows:
            total.update({k: v for k, v in r.items() if isinstance(v, int)})
            total['targets'] += len(r['targets'])
            total['targets_broken_run'] += sum(t['need'] is not None and t['longest'] >= t['need'] for t in r['targets'])
            total['beacons'] += sum(t['block'] == 'beacon' for t in r['targets'])
            total['beacons_broken_run'] += sum(t['block'] == 'beacon' and t['longest'] >= 90 for t in r['targets'])
        report[arm] = dict(runs=rows, total=dict(total))
        for r in rows:
            rate = r['target_attack'] / r['target'] if r['target'] else 0.
            print('%s %s dig %d | on target %d attack %.2f | other solid %d attack %d | air %d attack %d | %s' % (
                arm, r['run'], r['dig'], r['target'], rate, r['other'], r['other_attack'], r['air'], r['air_attack'],
                '; '.join('%s need %s longest %d bursts %d' % (t['block'], t['need'], t['longest'], t['bursts'])
                          for t in r['targets'])))
        t = total
        print('%s TOTAL on target %d/%d attack (%.3f), other solid %d/%d attack, targets with a long enough run %d/%d, '
              'beacons %d/%d' % (arm, t['target_attack'], t['target'], t['target_attack'] / max(t['target'], 1),
                                 t['other_attack'], t['other'], t['targets_broken_run'], t['targets'],
                                 t['beacons_broken_run'], t['beacons']))
    if args.json: args.json.write_bytes((json.dumps(report, indent=1) + '\n').encode('utf-8'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
