"""Building batches (design docs/specs/2026-11-14-build-design.md §4).

collect: one supervised game per (task, world seed, sampling seed) with `minecraftrl.pipeline.cli probe` on the blueprint
fixture and the opt-in place monitor, one policy step that ends when the server placement record covers the blueprint
(success), 60 s. The arm sets the build head: `--checkpoint official` is the untrained head (the official STEVE-1 output,
still recording latents). --teacher-check runs the preparation-only teacher instead (recorded as a scripted probe, never
a policy result, no head). Task ids and world seeds are checked against the mode (configs/build-v1.json): train =
training tasks on the training seeds, dev = dev tasks on 130363, final = final tasks on 662607. Finished runs are
skipped; <batch>/STOP halts; inputs are hashed into <batch>/frozen-<arm>-c<task>.json and must not change; an invalid
launch is moved to <batch>/excluded/ and halts the batch. Runs are <arm>/c<task>-s<seed>-n<n>.
--teacher recover (design docs/specs/2026-11-15-build-recover-design.md) names the recovering teacher in the fixture
(same task and world): the game then uses its stages, feed and teacher check, and its module is in the input record.
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
from minecraftrl.build_recover import build2_fixture  # noqa: E402
from minecraftrl.build_task import build_fixture, generate_build  # noqa: E402
from run_reacquire_batch import POLICY_DEVICE_MIB, frozen_ok, input_hashes, nvidia_free_mib, run_game, write  # noqa: E402

MODES = {'train': (271828, 314159, 161803), 'dev': (130363,), 'final': (662607,)}
REGISTRY = 'configs/build-v1.json'
BASE = 'configs/reacquire-v1.json'
SKILL = 'build'
LIMIT_S, FRAME_EVERY = 60, 2
TEACHERS = {'round1': build_fixture, 'recover': build2_fixture}


def load_registry(root): return json.loads((Path(root) / REGISTRY).read_text(encoding='utf-8'))


def check(mode, tasks, seeds, registry):
    bad = [s for s in seeds if s not in MODES[mode]]
    if bad: raise ValueError('seeds %s are not allowed in %s mode (allowed: %s)' % (bad, mode, list(MODES[mode])))
    allowed = registry['tasks'][mode]
    bad = [t for t in tasks if t not in allowed]
    if bad: raise ValueError('tasks %s are not %s tasks (%s)' % (bad, mode, allowed))


def build_config(base, registry, checkpoint, use_stage, teacher_check=False):
    """The reacquire config without any mining head, with the place monitor; a policy arm adds the build skill."""
    config = json.loads(json.dumps(base))
    for spec in config['skills'].values(): spec.pop('reacquire', None)
    config['place_monitor'] = True
    if not teacher_check:
        config['skills'][SKILL] = dict(embedding=registry['embedding'], sha256=registry['sha256'],
                                       cond_scale=float(registry['cond_scale']), source='text_prompt:' + registry['prompt'],
                                       reacquire=dict(checkpoint=checkpoint, use_stage=bool(use_stage), record_latents=True,
                                                      stages='build'))
    config['description'] = 'Building run (design 2026-11-14): ' + ('teacher check, no policy step' if teacher_check
                                                                          else 'build skill with the build head')
    return config


def prepare(batch, arm, root, task, checkpoint, use_stage, teacher_check=False, teacher='round1'):
    if teacher not in TEACHERS: raise ValueError('teacher must be one of %s' % sorted(TEACHERS))
    registry = load_registry(root)
    base = json.loads((Path(root) / BASE).read_text(encoding='utf-8'))
    config = write(batch / 'configs' / (arm + '.json'), build_config(base, registry, checkpoint, use_stage, teacher_check))
    fixture = write(batch / 'fixtures' / ('task-%d.json' % task), TEACHERS[teacher](generate_build(task)))
    step = (dict(id='check', skill='build_teacher', kind='scripted', target='build_done', count=1,
                 params=dict(driver='build_teacher', limit_steps=LIMIT_S * 20)) if teacher_check
            else dict(id='build', skill=SKILL, kind='policy', target='build_done', count=1, params=dict(until='build')))
    plan = write(batch / 'plans' / ('teacher.json' if teacher_check else 'build.json'), [step])
    return config, fixture, plan


def run_name(task, seed, n): return 'c%d-s%d-n%d' % (task, seed, n)


def collect(*, batch, arm, checkpoint, use_stage, mode, tasks, seeds, sampling, teacher_check=False, root=ROOT,
            launch=run_game, gpu_free=nvidia_free_mib, gpu_wait=(40, 15.), teacher='round1'):
    """0 done, 1 STOP, 3 GPU busy, 4 invalid launch excluded, 5 inputs changed."""
    root = Path(root).resolve(); batch = Path(batch).resolve(); registry = load_registry(root)
    check(mode, tasks, seeds, registry)
    if not batch.is_relative_to(root / 'results'): raise ValueError('batch must be under results/')
    if teacher_check and checkpoint is not None: raise ValueError('the teacher check has no head')
    if not teacher_check and not registry['sha256']: raise ValueError('the build prompt embedding is not made yet (%s)' % REGISTRY)
    if checkpoint is not None and not Path(checkpoint).exists(): raise ValueError('no checkpoint: %s' % checkpoint)
    meta = dict(arm=arm, checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()), use_stage=bool(use_stage),
                mode=mode, limit_s=LIMIT_S, frame_every=FRAME_EVERY, teacher_check=bool(teacher_check),
                scope='building; no API; success = the server placement record covers the blueprint')
    if teacher != 'round1': meta['teacher'] = teacher          # round 1 records stay as they were
    for task in tasks:
        config, fixture, plan = prepare(batch, arm, root, task, meta['checkpoint'], use_stage, teacher_check, teacher)
        files = [config, fixture, plan, root / REGISTRY, root / 'src/minecraftrl/build_task.py',
                 root / 'src/minecraftrl/place_task.py',
                 root / 'src/minecraftrl/PlaceObservation.java', root / 'src/minecraftrl/world_save_patch.py']
        if teacher != 'round1': files.append(root / 'src/minecraftrl/build_recover.py')
        if not teacher_check: files.append(root / registry['embedding'])
        if checkpoint: files.append(Path(checkpoint))
        frozen = batch / ('frozen-%s-c%d.json' % (arm, task))
        for seed in seeds:
            for n in sampling:
                run_dir = batch / arm / run_name(task, seed, n)
                if (run_dir / 'summary.json').exists(): continue
                if (batch / 'STOP').exists(): print('STOP present; batch halted', flush=True); return 1
                if not frozen_ok(frozen, input_hashes(root, files), dict(meta, task=task)): return 5
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
                line = dict(arm=arm, task=task, run=run_dir.name, seed=seed, sampling=n, status=summary.get('status'),
                            execution_valid=summary.get('execution_valid'),
                            success=bool(summary.get('execution_valid') and summary.get('all_steps_completed')),
                            step_status=stage.get('status'), filled=stage.get('filled'), wrong=stage.get('wrong'),
                            unknown_placements=stage.get('unknown_placements'), step_steps=stage.get('steps'),
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
    p.add_argument('--tasks', type=int, nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--sampling', type=int, nargs='+', required=True)
    p.add_argument('--teacher-check', action='store_true', help='preparation-only teacher run (design §5), no policy')
    p.add_argument('--teacher', choices=tuple(TEACHERS), default='round1', help='the teacher named in the fixture')
    args = p.parse_args(argv)
    if not args.teacher_check and args.checkpoint is None: p.error('--checkpoint is required for a policy arm')
    checkpoint = None if args.checkpoint in (None, 'official') else Path(args.checkpoint)
    return collect(batch=args.batch, arm=args.arm, checkpoint=checkpoint, use_stage=args.use_stage == 'on', mode=args.mode,
                   tasks=args.tasks, seeds=args.seeds, sampling=args.sampling, teacher_check=args.teacher_check,
                   teacher=args.teacher)


if __name__ == '__main__':
    sys.exit(main())
