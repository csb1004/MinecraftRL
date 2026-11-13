"""Block placing batches (design docs/specs/2026-11-14-place-block-design.md §5).

collect: one supervised game per (target, world seed, sampling seed) with `minecraftrl.pipeline.cli probe` on the target
fixture and the opt-in place monitor, one policy step that ends on a server placement at the target (success), 45 s.
The arm sets the place head: `--checkpoint official` is the untrained head (the official STEVE-1 output, still recording
latents). --teacher-check runs the preparation-only teacher instead (recorded as a scripted probe, never a policy
result, no head). Target ids and world seeds are checked against the mode (configs/place-v1.json): train = training
targets on the training seeds, dev = dev targets on 130363, final = final targets on 357913. Finished runs are skipped;
<batch>/STOP halts; inputs are hashed into <batch>/frozen-<arm>-t<target>.json and must not change; an invalid launch is
moved to <batch>/excluded/ and halts the batch. Runs are <arm>/t<target>-s<seed>-n<n>.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from minecraftrl.place_task import generate_target, place_fixture  # noqa: E402
from run_reacquire_batch import POLICY_DEVICE_MIB, frozen_ok, input_hashes, nvidia_free_mib, run_game, write  # noqa: E402

MODES = {'train': (271828, 314159, 161803), 'dev': (130363,), 'final': (357913,)}
REGISTRY = 'configs/place-v1.json'
BASE = 'configs/reacquire-v1.json'
SKILL = 'place_block'
LIMIT_S, FRAME_EVERY = 45, 2


def load_registry(root): return json.loads((Path(root) / REGISTRY).read_text(encoding='utf-8'))


def check(mode, targets, seeds, registry):
    bad = [s for s in seeds if s not in MODES[mode]]
    if bad: raise ValueError('seeds %s are not allowed in %s mode (allowed: %s)' % (bad, mode, list(MODES[mode])))
    allowed = registry['targets'][mode]
    bad = [t for t in targets if t not in allowed]
    if bad: raise ValueError('targets %s are not %s targets (%s)' % (bad, mode, allowed))


def place_config(base, registry, checkpoint, use_stage, teacher_check=False):
    """The reacquire config without any mining head, with the place monitor; a policy arm adds the place skill."""
    config = json.loads(json.dumps(base))
    for spec in config['skills'].values(): spec.pop('reacquire', None)
    config['place_monitor'] = True
    if not teacher_check:
        config['skills'][SKILL] = dict(embedding=registry['embedding'], sha256=registry['sha256'],
                                       cond_scale=float(registry['cond_scale']), source='text_prompt:' + registry['prompt'],
                                       reacquire=dict(checkpoint=checkpoint, use_stage=bool(use_stage), record_latents=True,
                                                      stages='place'))
    config['description'] = 'Block placing run (design 2026-11-14): ' + ('teacher check, no policy step' if teacher_check
                                                                          else 'place_block skill with the place head')
    return config


def prepare(batch, arm, root, target, checkpoint, use_stage, teacher_check=False):
    registry = load_registry(root)
    base = json.loads((Path(root) / BASE).read_text(encoding='utf-8'))
    config = write(batch / 'configs' / (arm + '.json'), place_config(base, registry, checkpoint, use_stage, teacher_check))
    fixture = write(batch / 'fixtures' / ('target-%d.json' % target), place_fixture(generate_target(target)))
    step = (dict(id='check', skill='place_teacher', kind='scripted', target='place_target', count=1,
                 params=dict(driver='place_teacher', limit_steps=LIMIT_S * 20)) if teacher_check
            else dict(id='place', skill=SKILL, kind='policy', target='place_target', count=1, params=dict(until='place')))
    plan = write(batch / 'plans' / ('teacher.json' if teacher_check else 'place.json'), [step])
    return config, fixture, plan


def run_name(target, seed, n): return 't%d-s%d-n%d' % (target, seed, n)


def collect(*, batch, arm, checkpoint, use_stage, mode, targets, seeds, sampling, teacher_check=False, root=ROOT,
            launch=run_game, gpu_free=nvidia_free_mib, gpu_wait=(40, 15.)):
    """0 done, 1 STOP, 3 GPU busy, 4 invalid launch excluded, 5 inputs changed."""
    root = Path(root).resolve(); batch = Path(batch).resolve(); registry = load_registry(root)
    check(mode, targets, seeds, registry)
    if not batch.is_relative_to(root / 'results'): raise ValueError('batch must be under results/')
    if teacher_check and checkpoint is not None: raise ValueError('the teacher check has no head')
    if not teacher_check and not registry['sha256']: raise ValueError('the place prompt embedding is not made yet (%s)' % REGISTRY)
    if checkpoint is not None and not Path(checkpoint).exists(): raise ValueError('no checkpoint: %s' % checkpoint)
    meta = dict(arm=arm, checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()), use_stage=bool(use_stage),
                mode=mode, limit_s=LIMIT_S, frame_every=FRAME_EVERY, teacher_check=bool(teacher_check),
                scope='block placing; no API; success = a server placement at the target cell')
    for target in targets:
        config, fixture, plan = prepare(batch, arm, root, target, meta['checkpoint'], use_stage, teacher_check)
        files = [config, fixture, plan, root / REGISTRY, root / 'src/minecraftrl/place_task.py',
                 root / 'src/minecraftrl/PlaceObservation.java', root / 'src/minecraftrl/world_save_patch.py']
        if not teacher_check: files.append(root / registry['embedding'])
        if checkpoint: files.append(Path(checkpoint))
        frozen = batch / ('frozen-%s-t%d.json' % (arm, target))
        for seed in seeds:
            for n in sampling:
                run_dir = batch / arm / run_name(target, seed, n)
                if (run_dir / 'summary.json').exists(): continue
                if (batch / 'STOP').exists(): print('STOP present; batch halted', flush=True); return 1
                if not frozen_ok(frozen, input_hashes(root, files), dict(meta, target=target)): return 5
                tries, pause = gpu_wait; free = gpu_free()
                for _ in range(tries - 1):
                    if free >= POLICY_DEVICE_MIB: break
                    time.sleep(pause); free = gpu_free()
                if free < POLICY_DEVICE_MIB: print('GPU busy: %.0f MiB free; batch halted' % free, flush=True); return 3
                command = [sys.executable, '-m', 'minecraftrl.pipeline.cli', 'probe', '--root', str(root),
                           '--output', str(run_dir), '--config', str(config), '--plan-json', str(plan), '--fixture', str(fixture),
                           '--stop-file', str(batch / 'STOP'), '--seed', str(seed), '--sampling-seed', str(n),
                           '--limit-s', str(LIMIT_S), '--frame-every', str(FRAME_EVERY)]
                print('%s %s: start %s' % (arm, run_dir.name, time.strftime('%H:%M:%S')), flush=True)
                launch(command, run_dir)
                summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8')) if (run_dir / 'summary.json').exists() \
                    else dict(status='missing_summary', execution_valid=False)
                stage = (summary.get('stage_results') or [{}])[-1]
                line = dict(arm=arm, target=target, run=run_dir.name, seed=seed, sampling=n, status=summary.get('status'),
                            execution_valid=summary.get('execution_valid'),
                            success=bool(summary.get('execution_valid') and summary.get('all_steps_completed')),
                            step_status=stage.get('status'), wrong=stage.get('wrong'), step_steps=stage.get('steps'),
                            release=summary.get('release'),
                            remaining_pids=(summary.get('process_cleanup') or {}).get('remaining_live_pids'))
                if not summary.get('execution_valid'):
                    target_dir = batch / 'excluded' / arm / ('%s-%d' % (run_dir.name, int(time.time())))
                    target_dir.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(run_dir), str(target_dir))
                    line.update(excluded=True, moved_to=str(target_dir.relative_to(batch)))
                with (batch / 'results.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(line) + '\n')
                print(json.dumps(line), flush=True)
                if line.get('excluded'): print('invalid launch excluded; batch halted', flush=True); return 4
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--checkpoint', help="path, or 'official' for the untrained head; omit with --teacher-check")
    p.add_argument('--use-stage', choices=('on', 'off'), default='on')
    p.add_argument('--mode', choices=tuple(MODES), required=True)
    p.add_argument('--targets', type=int, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--sampling', type=int, nargs='+', required=True)
    p.add_argument('--teacher-check', action='store_true', help='preparation-only teacher run (design §5), no policy')
    args = p.parse_args(argv)
    if not args.teacher_check and args.checkpoint is None: p.error('--checkpoint is required for a policy arm')
    checkpoint = None if args.checkpoint in (None, 'official') else Path(args.checkpoint)
    return collect(batch=args.batch, arm=args.arm, checkpoint=checkpoint, use_stage=args.use_stage == 'on', mode=args.mode,
                   targets=args.targets, seeds=args.seeds, sampling=args.sampling, teacher_check=args.teacher_check)


if __name__ == '__main__':
    sys.exit(main())
