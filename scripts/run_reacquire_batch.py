"""Reacquire probe batches (design 2026-11-07 §6-§7): collect runs per arm, then compare arms.

collect: one supervised game per (world seed, sampling seed) with `minecraftrl.pipeline.cli probe`, in the scene of the
2026-11-04 iron probe (fixture v2 + a given stone pickaxe, one iron step, 180 s, tool lock, no oversight). The arm sets
the reacquire head: `--checkpoint official` is the untrained head (the official STEVE-1 output, still recording
latents). Finished runs are skipped; <batch>/STOP halts; inputs are hashed into <batch>/frozen-<arm>.json and must not
change; an invalid launch is moved to <batch>/excluded/ and halts the batch. Seeds are checked against the mode:
train = training seeds only, dev = 130363, final = 577215.

compare: success, iron-step game seconds and skill-stage shares per arm, plus the pre-set verdict against `official`.

--block (multi-block design 2026-11-08 §3): a training block from configs/multi-block-v1.json replaces the two iron ores
of the arena; the step mines that block with its own "mine <block>" skill carrying the reacquire head. Without --block
the files, run folders (s<seed>-n<n>) and input record (frozen-<arm>.json) are exactly the iron probe's; a block run is
<arm>/<block>-s<seed>-n<n> with frozen-<arm>-<block>.json.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.reacquire_data import TRAIN_SEEDS  # noqa: E402

MODES = {'train': tuple(TRAIN_SEEDS), 'dev': (130363,), 'final': (577215,)}
SKILL = 'mine_iron_reacquire'
IRON = 'iron_ore'
REGISTRY = 'configs/multi-block-v1.json'
POLICY_DEVICE_MIB = 2150
LIMIT_S, FRAME_EVERY = 180, 2
WATCHED = ('configs/reacquire-v1.json', 'configs/iron-demo-fixture-v2.json', 'src/minecraftrl/reacquire_head.py',
           'src/minecraftrl/workshop.py', 'src/minecraftrl/workshop_recipes.py', 'src/minecraftrl/DemoFixture.java',
           'src/minecraftrl/DemoObservation.java', 'src/minecraftrl/WorkshopObservation.java', 'src/minecraftrl/policies.py')


def check_seeds(mode, seeds):
    allowed = MODES[mode]
    bad = [s for s in seeds if s not in allowed]
    if bad: raise ValueError('seeds %s are not allowed in %s mode (allowed: %s)' % (bad, mode, list(allowed)))


def arm_config(base, checkpoint, use_stage):
    config = json.loads(json.dumps(base))
    config['skills'][SKILL]['reacquire'] = dict(checkpoint=checkpoint, use_stage=bool(use_stage), record_latents=True)
    return config


def load_registry(root):
    return json.loads((Path(root) / REGISTRY).read_text(encoding='utf-8'))


def block_entry(registry, block):
    if block not in registry['blocks']:
        raise ValueError('unknown training block %s (registry: %s)' % (block, ', '.join(sorted(registry['blocks']))))
    return registry['blocks'][block]


def block_config(base, registry, block, checkpoint, use_stage):
    """The iron config with one more text skill for `block`; only that skill carries the reacquire head."""
    entry = block_entry(registry, block)
    config = json.loads(json.dumps(base))
    for spec in config['skills'].values(): spec.pop('reacquire', None)
    config['skills'][entry['skill']] = dict(
        embedding=entry['embedding'], sha256=entry['sha256'], cond_scale=float(registry['cond_scale']),
        source='text_prompt:' + entry['prompt'], target_block=block, target_items=[entry['item']],
        reacquire=dict(checkpoint=checkpoint, use_stage=bool(use_stage), record_latents=True))
    return config


def block_fixture(base, block):
    """Fixture v2 with its iron ores replaced by `block`; fills, trees and every other block stay."""
    return dict(base, blocks=[dict(b, block=block) if b['block'] == IRON else b for b in base['blocks']])


def run_name(block, seed, n):
    return ('s%d-n%d' if block == IRON else block + '-s%d-n%d') % (seed, n)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return path


def prepare(batch, arm, root, checkpoint, use_stage, block=IRON):
    base = json.loads((root / 'configs/reacquire-v1.json').read_text(encoding='utf-8'))
    fixture_base = json.loads((root / 'configs/iron-demo-fixture-v2.json').read_text(encoding='utf-8'))
    if block != IRON:
        registry = load_registry(root); entry = block_entry(registry, block)
        config = write(batch / 'configs' / ('%s-%s.json' % (arm, block)), block_config(base, registry, block, checkpoint, use_stage))
        fixture = write(batch / 'fixtures' / (block + '-probe.json'), dict(
            block_fixture(fixture_base, block), give=[dict(item='stone_pickaxe', count=1)],
            description=fixture_base['description'] + ' DEVELOPMENT PROBE VARIANT: start items given by fixture.'
            + ' MULTI-BLOCK VARIANT (design 2026-11-08): the iron ores are replaced by %s.' % block))
        plan = write(batch / 'plans' / (block + '.json'), [dict(id=block, skill=entry['skill'], kind='policy',
                                                                target=entry['item'], count=1, block=block)])
        return config, fixture, plan
    config = write(batch / 'configs' / (arm + '.json'), arm_config(base, checkpoint, use_stage))
    fixture = write(batch / 'fixtures/iron-probe.json', dict(fixture_base, give=[dict(item='stone_pickaxe', count=1)],
        description=fixture_base['description'] + ' DEVELOPMENT PROBE VARIANT: start items given by fixture.'))
    plan = write(batch / 'plans/iron.json', [dict(id='iron', skill=SKILL, kind='policy', target='iron_ore', count=1,
                                                  block='iron_ore')])
    return config, fixture, plan


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(root, files):
    names = list(WATCHED) + sorted(str(p.relative_to(root)) for p in (root / 'src/minecraftrl/pipeline').glob('*.py'))
    hashes = {n: sha256(root / n) for n in names if (root / n).exists()}
    hashes.update({str(Path(f).resolve()): sha256(f) for f in files})
    return hashes


def frozen_ok(path, hashes, meta):
    """Write the input record on first use; afterwards every input must be unchanged."""
    if not path.exists(): write(path, dict(hashes=hashes, **meta)); return True
    old = json.loads(path.read_text(encoding='utf-8'))
    changed = sorted(k for k in set(old['hashes']) | set(hashes) if old['hashes'].get(k) != hashes.get(k))
    if changed: print('INPUTS CHANGED: ' + ', '.join(changed), flush=True)
    return not changed


def nvidia_free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', '-i', '0'],
                         capture_output=True, text=True, timeout=10)
    used, total = (float(v) for v in out.stdout.strip().split(','))
    return total - used


def run_game(command, run_dir):
    subprocess.run(command, cwd=str(ROOT), check=False)


def collect(*, batch, arm, checkpoint, use_stage, seeds, sampling, mode='train', root=ROOT, launch=run_game,
            gpu_free=nvidia_free_mib, gpu_wait=(40, 15.), block=IRON):
    """0 done, 1 STOP, 3 GPU busy, 4 invalid launch excluded, 5 inputs changed."""
    check_seeds(mode, seeds)
    root = Path(root).resolve(); batch = Path(batch).resolve()
    if not batch.is_relative_to(root / 'results'): raise ValueError('batch must be under results/')
    if checkpoint is not None and not Path(checkpoint).exists(): raise ValueError('no checkpoint: %s' % checkpoint)
    config, fixture, plan = prepare(batch, arm, root, None if checkpoint is None else str(Path(checkpoint).resolve()), use_stage,
                                    block=block)
    skill = SKILL if block == IRON else block_entry(load_registry(root), block)['skill']
    embedding = root / json.loads(config.read_text(encoding='utf-8'))['skills'][skill]['embedding']
    files = [config, fixture, plan, embedding] + ([checkpoint] if checkpoint else []) + ([] if block == IRON else [root / REGISTRY])
    frozen = batch / ('frozen-%s.json' % arm if block == IRON else 'frozen-%s-%s.json' % (arm, block))
    meta = dict(arm=arm, checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()), use_stage=bool(use_stage),
                mode=mode, limit_s=LIMIT_S, frame_every=FRAME_EVERY, scope='reacquire probe; no API; success = server inventory')
    if block != IRON: meta['block'] = block
    for seed in seeds:
        for n in sampling:
            run_dir = batch / arm / run_name(block, seed, n)
            if (run_dir / 'summary.json').exists(): continue
            if (batch / 'STOP').exists(): print('STOP present; batch halted', flush=True); return 1
            if not frozen_ok(frozen, input_hashes(root, files), meta): return 5
            tries, pause = gpu_wait; free = gpu_free()
            for _ in range(tries - 1):
                if free >= POLICY_DEVICE_MIB: break
                time.sleep(pause); free = gpu_free()
            if free < POLICY_DEVICE_MIB: print('GPU busy: %.0f MiB free; batch halted' % free, flush=True); return 3
            command = [sys.executable, '-m', 'minecraftrl.pipeline.cli', 'probe', '--root', str(root), '--output', str(run_dir),
                       '--config', str(config), '--plan-json', str(plan), '--fixture', str(fixture),
                       '--stop-file', str(batch / 'STOP'), '--seed', str(seed), '--sampling-seed', str(n),
                       '--limit-s', str(LIMIT_S), '--frame-every', str(FRAME_EVERY)]
            print('%s %s: start %s' % (arm, run_dir.name, time.strftime('%H:%M:%S')), flush=True)
            launch(command, run_dir)
            summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8')) if (run_dir / 'summary.json').exists() \
                else dict(status='missing_summary', execution_valid=False)
            line = dict(arm=arm, block=block, run=run_dir.name, seed=seed, sampling=n, status=summary.get('status'),
                        execution_valid=summary.get('execution_valid'),
                        success=bool(summary.get('execution_valid') and summary.get('all_steps_completed')),
                        steps=summary.get('steps'), elapsed_control_s=summary.get('elapsed_control_s'),
                        release=summary.get('release'),
                        remaining_pids=(summary.get('process_cleanup') or {}).get('remaining_live_pids'))
            if not summary.get('execution_valid'):
                target = batch / 'excluded' / arm / ('%s-%d' % (run_dir.name, int(time.time())))
                target.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(run_dir), str(target))
                line.update(excluded=True, moved_to=str(target.relative_to(batch)))
            with (batch / 'results.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(line) + '\n')
            print(json.dumps(line), flush=True)
            if line.get('excluded'): print('invalid launch excluded; batch halted', flush=True); return 4
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--batch', type=Path, required=True)
    c.add_argument('--arm', required=True)
    c.add_argument('--checkpoint', required=True, help="path, or 'official' for the untrained head")
    c.add_argument('--use-stage', choices=('on', 'off'), required=True)
    c.add_argument('--seeds', type=int, nargs='+', required=True)
    c.add_argument('--sampling', type=int, nargs='+', required=True)
    c.add_argument('--mode', choices=tuple(MODES), default='train')
    c.add_argument('--block', default=IRON, help='a training block of %s; default: the iron probe' % REGISTRY)
    k = sub.add_parser('compare')
    k.add_argument('--batch', type=Path, required=True)
    k.add_argument('--arms', nargs='+', required=True)
    k.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'collect':
        checkpoint = None if args.checkpoint == 'official' else Path(args.checkpoint)
        return collect(batch=args.batch, arm=args.arm, checkpoint=checkpoint, use_stage=args.use_stage == 'on',
                       seeds=args.seeds, sampling=args.sampling, mode=args.mode, block=args.block)
    from minecraftrl.reacquire_compare import compare, report_markdown
    batch = args.batch.resolve()
    arms = {arm: sorted(d for d in (batch / arm).iterdir() if (d / 'summary.json').exists()) for arm in args.arms}
    report = compare(arms, root=ROOT)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    write(out / 'report.json', report)
    (out / 'report.md').write_bytes(report_markdown(report).encode('utf-8'))
    print(report_markdown(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
