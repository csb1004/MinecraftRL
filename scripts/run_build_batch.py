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
Mode occluded (occlusion check, docs/specs/2026-11-15-build-occluded-check-design.md): occluded tasks on 707106, the
recovering teacher only; the fixture sets a block right in front of the first cell (build_recover.occluded_build).
Round 4 (docs/specs/2026-11-15-build-careful-design.md): --teacher careful (careful place); modes occluded_train
(occluded training tasks on the training seeds), dev4 / dev4_occluded (302585) and final4 / final4_occluded (866025).
Every occluded mode needs a recovering teacher (recover or careful).
Mode large (size check, docs/specs/2026-11-15-build-large-check-design.md): larger blueprints (build_large.large_build,
6-16 cells) on 693147, a recovering teacher only; build_large.py joins the input record.
Round 5 (docs/specs/2026-11-15-build-repair-design.md): modes repair_train (training seeds), repair_dev (414213) and
repair_final (236067) run repair tasks (build_repair.repair_build: stray blocks next to the blueprint, a stone pickaxe in
slot 2) with --teacher repair (success = blueprint filled and no stray left) or --teacher careful (the round 4 skill as a
reference; its game ends when the blueprint is filled). The config lets the policy press hotbar.1 and hotbar.2 through
the tool lock; build_repair.py joins the input record.
Mode beacon (demo, user request 2026-11-23): the beacon pyramid (build_beacon.beacon_build: gold, iron and diamond blocks
and a beacon, 10 cells) on 316227 with the sensor teacher only. The session's rule selects the hotbar slot of the
current cell's block (recorded as rule_select); a block of another kind in a blueprint cell is wrong. After success the
game keeps neutral input for HOLD_AFTER_STEPS steps so the frames show the beam. build_beacon.py joins the input record.
Sensor repair (user request 2026-11-23, docs/specs/2026-11-23-sensor-repair-design.md): modes dig_train (training seeds)
and dig_dev (331662) run dig tasks (build_sensor_repair.dig_task: wrong blocks to dig and pick up) with --teacher
sensor_dig, the dig head alone; mode beacon_repair runs the beacon pyramid on 316227 with --teacher sensor_repair:
exact gifts and a diamond pickaxe, the building head (--checkpoint) and the dig head (--dig-checkpoint) switched by the
anomaly monitor, REPAIR_LIMIT_S. The session's rule selects the needed item from the client hand (rule_select).
build_sensor_repair.py and head_switch.py join the input record.
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
from minecraftrl.build_beacon import beacon_build  # noqa: E402
from minecraftrl.build_large import large_build  # noqa: E402
from minecraftrl.build_recover import CAREFUL, REPAIR, build2_fixture, occluded_build  # noqa: E402
from minecraftrl.build_repair import repair_build, repair_fixture  # noqa: E402
from minecraftrl.build_sensor import blueprint_of, sensor_fixture  # noqa: E402
from minecraftrl.build_sensor_repair import SENSOR_DIG, SENSOR_REPAIR, dig_task  # noqa: E402
from minecraftrl.build_sensor_repair import repair_fixture as sensor_repair_fixture  # noqa: E402
from minecraftrl.build_task import build_fixture, generate_build  # noqa: E402
from run_reacquire_batch import POLICY_DEVICE_MIB, frozen_ok, input_hashes, nvidia_free_mib, run_game, write  # noqa: E402

TRAIN_SEEDS = (271828, 314159, 161803)
MODES = {'train': TRAIN_SEEDS, 'dev': (130363,), 'final': (662607,), 'occluded': (707106,),
         'occluded_train': TRAIN_SEEDS, 'dev4': (302585,), 'dev4_occluded': (302585,), 'final4': (866025,),
         'final4_occluded': (866025,), 'large': (693147,), 'repair_train': TRAIN_SEEDS, 'repair_dev': (414213,),
         'repair_final': (236067,), 'sensor_final': (618034,), 'beacon': (316227,), 'dig_train': TRAIN_SEEDS,
         'dig_dev': (331662,), 'beacon_repair': (316227,)}
OCCLUDED_MODES = ('occluded', 'occluded_train', 'dev4_occluded', 'final4_occluded')
LARGE_MODES = ('large',)
BEACON_MODES = ('beacon',)
HOLD_AFTER_STEPS = 160   # neutral steps after a beacon success: the beam turns on within 80 game ticks
BEACON_LIMIT_S = 90      # beacon games: 10 cells, slot changes and the hold after success
DIG_MODES = ('dig_train', 'dig_dev')
BEACON_REPAIR_MODES = ('beacon_repair',)
REPAIR_LIMIT_S = 120     # beacon games with repairs: digs and pickups on top of the 10 cells
SENSOR_TEACHERS = ('sensor', SENSOR_DIG, SENSOR_REPAIR)
ATTACK_DECODES = ('sample', 'threshold')   # the dig head's attack key: sampled, or from its own probability
REPAIR_MODES = ('repair_train', 'repair_dev', 'repair_final')
RECOVERING = ('recover', CAREFUL, 'sensor')
REPAIRING = (REPAIR, CAREFUL)            # teachers that can run a repair task (careful: the round 4 reference)
REPAIR_SLOTS = ('hotbar.1', 'hotbar.2')  # the block and the stone pickaxe pass the policy tool lock in repair tasks
REGISTRY = 'configs/build-v1.json'
BASE = 'configs/reacquire-v1.json'
SKILL = 'build'
LIMIT_S, FRAME_EVERY = 60, 2
TEACHERS = {'round1': build_fixture, 'recover': build2_fixture, CAREFUL: lambda task: build2_fixture(task, teacher=CAREFUL),
            REPAIR: repair_fixture, 'sensor': sensor_fixture, SENSOR_DIG: lambda task: sensor_repair_fixture(task, SENSOR_DIG),
            SENSOR_REPAIR: lambda task: sensor_repair_fixture(task, SENSOR_REPAIR)}


def load_registry(root): return json.loads((Path(root) / REGISTRY).read_text(encoding='utf-8'))


def check(mode, tasks, seeds, registry):
    bad = [s for s in seeds if s not in MODES[mode]]
    if bad: raise ValueError('seeds %s are not allowed in %s mode (allowed: %s)' % (bad, mode, list(MODES[mode])))
    allowed = registry['tasks'][mode]
    bad = [t for t in tasks if t not in allowed]
    if bad: raise ValueError('tasks %s are not %s tasks (%s)' % (bad, mode, allowed))


def build_config(base, registry, checkpoint, use_stage, teacher_check=False, tool_allowed=(), sensor=False,
                 dig_checkpoint=None, attack_decode='sample'):
    """The reacquire config without any mining head, with the place monitor; a policy arm adds the build skill.
    `tool_allowed`: hotbar keys the policy may press through the tool lock (repair tasks). `dig_checkpoint`: the dig
    head next to the building head (sensor repair). `attack_decode` 'threshold': the dig head's attack key from its own
    probability (comparison arm, user decision 2026-11-24); 'sample' leaves the config as before."""
    if attack_decode not in ATTACK_DECODES: raise ValueError('attack_decode must be one of %s' % (ATTACK_DECODES,))
    config = json.loads(json.dumps(base))
    for spec in config['skills'].values(): spec.pop('reacquire', None)
    config['place_monitor'] = True
    if sensor: config['sensor_monitor'] = True    # client sensors for the sensor teacher (plan 2026-11-22)
    if tool_allowed: config['policy_tool_allowed'] = list(tool_allowed)
    if not teacher_check:
        config['skills'][SKILL] = dict(embedding=registry['embedding'], sha256=registry['sha256'],
                                       cond_scale=float(registry['cond_scale']), source='text_prompt:' + registry['prompt'],
                                       reacquire=dict(checkpoint=checkpoint, use_stage=bool(use_stage), record_latents=True,
                                                      stages='build'))
        if dig_checkpoint is not None: config['skills'][SKILL]['reacquire']['dig_checkpoint'] = dig_checkpoint
        if attack_decode != 'sample': config['skills'][SKILL]['reacquire']['attack_decode'] = attack_decode
    config['description'] = 'Building run (design 2026-11-14): ' + ('teacher check, no policy step' if teacher_check
                                                                          else 'build skill with the build head')
    return config


def prepare(batch, arm, root, task, checkpoint, use_stage, teacher_check=False, teacher='round1', occluded=False,
            large=False, repair=False, beacon=False, dig=False, beacon_repair=False, dig_checkpoint=None,
            attack_decode='sample'):
    if teacher not in TEACHERS: raise ValueError('teacher must be one of %s' % sorted(TEACHERS))
    if occluded and teacher not in RECOVERING: raise ValueError('occluded tasks need a recovering teacher')
    if large and teacher not in RECOVERING: raise ValueError('large tasks need a recovering teacher')
    if repair and teacher not in REPAIRING: raise ValueError('repair tasks need the repair or the careful teacher')
    if teacher == REPAIR and not repair: raise ValueError('the repair teacher runs repair tasks only')
    if beacon and teacher != 'sensor': raise ValueError('beacon tasks need the sensor teacher')
    if dig != (teacher == SENSOR_DIG): raise ValueError('dig tasks run with the sensor_dig teacher only, and it with them')
    if beacon_repair != (teacher == SENSOR_REPAIR):
        raise ValueError('beacon_repair runs with the sensor_repair teacher only, and it with it')
    if beacon_repair and not teacher_check and dig_checkpoint is None: raise ValueError('beacon_repair needs the dig head')
    blueprint = (occluded_build(task) if occluded else large_build(task) if large else repair_build(task) if repair
                 else beacon_build(task) if beacon or beacon_repair else dig_task(task) if dig else generate_build(task))
    if blueprint is None: raise ValueError('task %d has no free cell in front of its first cell' % task)
    registry = load_registry(root)
    base = json.loads((Path(root) / BASE).read_text(encoding='utf-8'))
    config = write(batch / 'configs' / (arm + '.json'), build_config(base, registry, checkpoint, use_stage, teacher_check,
                                                                     tool_allowed=REPAIR_SLOTS if repair else (),
                                                                     sensor=teacher in SENSOR_TEACHERS,
                                                                     dig_checkpoint=None if teacher_check else dig_checkpoint,
                                                                     attack_decode=attack_decode))
    if teacher in SENSOR_TEACHERS:     # the blueprint instruction the sensor feed reads, kept apart from the fixture's world
        write(batch / 'blueprints' / ('task-%d.json' % task), blueprint_of(blueprint))
    fixture = write(batch / 'fixtures' / ('task-%d.json' % task),
                    repair_fixture(blueprint, teacher=teacher) if repair else TEACHERS[teacher](blueprint))
    step = (dict(id='check', skill='build_teacher', kind='scripted', target='build_done', count=1,
                 params=dict(driver='build_teacher', limit_steps=limit_for(beacon, beacon_repair) * 20)) if teacher_check
            else dict(id='build', skill=SKILL, kind='policy', target='build_done', count=1, params=dict(until='build')))
    if beacon or beacon_repair: step['params']['hold_after_steps'] = HOLD_AFTER_STEPS
    plan = write(batch / 'plans' / ('teacher.json' if teacher_check else 'build.json'), [step])
    return config, fixture, plan


def run_name(task, seed, n): return 'c%d-s%d-n%d' % (task, seed, n)


def limit_for(beacon=False, beacon_repair=False):
    return REPAIR_LIMIT_S if beacon_repair else BEACON_LIMIT_S if beacon else LIMIT_S


def collect(*, batch, arm, checkpoint, use_stage, mode, tasks, seeds, sampling, teacher_check=False, root=ROOT,
            launch=run_game, gpu_free=nvidia_free_mib, gpu_wait=(40, 15.), teacher='round1', dig_checkpoint=None,
            attack_decode='sample'):
    """0 done, 1 STOP, 3 GPU busy, 4 invalid launch excluded, 5 inputs changed."""
    if attack_decode not in ATTACK_DECODES: raise ValueError('attack_decode must be one of %s' % (ATTACK_DECODES,))
    if attack_decode != 'sample' and (teacher_check or teacher not in (SENSOR_DIG, SENSOR_REPAIR)):
        raise ValueError('the attack decode needs a dig head (dig or beacon_repair policy arm)')
    root = Path(root).resolve(); batch = Path(batch).resolve(); registry = load_registry(root)
    check(mode, tasks, seeds, registry)
    if not batch.is_relative_to(root / 'results'): raise ValueError('batch must be under results/')
    if teacher_check and checkpoint is not None: raise ValueError('the teacher check has no head')
    if mode in OCCLUDED_MODES and teacher not in RECOVERING: raise ValueError('occluded tasks need a recovering teacher')
    if mode in LARGE_MODES and teacher not in RECOVERING: raise ValueError('large tasks need a recovering teacher')
    if mode in REPAIR_MODES and teacher not in REPAIRING: raise ValueError('repair tasks need the repair or the careful teacher')
    if teacher == REPAIR and mode not in REPAIR_MODES: raise ValueError('the repair teacher runs repair tasks only')
    if mode in BEACON_MODES and teacher != 'sensor': raise ValueError('beacon tasks need the sensor teacher')
    if (mode in DIG_MODES) != (teacher == SENSOR_DIG): raise ValueError('dig modes run with the sensor_dig teacher only')
    if (mode in BEACON_REPAIR_MODES) != (teacher == SENSOR_REPAIR):
        raise ValueError('beacon_repair runs with the sensor_repair teacher only')
    if teacher == SENSOR_REPAIR and not teacher_check and (checkpoint is None or dig_checkpoint is None):
        raise ValueError('beacon_repair needs the building head (--checkpoint) and the dig head (--dig-checkpoint)')
    if dig_checkpoint is not None and teacher != SENSOR_REPAIR: raise ValueError('only beacon_repair takes a dig head')
    if dig_checkpoint is not None and not Path(dig_checkpoint).exists(): raise ValueError('no checkpoint: %s' % dig_checkpoint)
    if not teacher_check and not registry['sha256']: raise ValueError('the build prompt embedding is not made yet (%s)' % REGISTRY)
    if checkpoint is not None and not Path(checkpoint).exists(): raise ValueError('no checkpoint: %s' % checkpoint)
    scope = 'building; no API; success = the server placement record covers the blueprint'
    if teacher == REPAIR:
        scope += ' and no stray left (server placement and destroy records); policy hotbar.1/hotbar.2 pass the tool lock'
    elif mode in REPAIR_MODES:
        scope += ' (repair task as a reference: strays are scored from the records); policy hotbar.1/hotbar.2 pass the tool lock'
    if mode in BEACON_MODES:
        scope += ('; multi-material: a session rule presses the hotbar key of the current cell\'s block (rule_select, not'
                  ' policy); a block of another kind in a blueprint cell is wrong; %d neutral steps after success'
                  % HOLD_AFTER_STEPS)
    if mode in DIG_MODES:
        scope = ('sensor repair dig tasks; no API; success = every wrong block destroyed and back in the server inventory '
                 '(server destroy record and inventory); a session rule presses the hotbar key of the pickaxe (rule_select, '
                 'not policy)')
    if mode in BEACON_REPAIR_MODES:
        scope = ('sensor repair beacon demo; no API; success = every blueprint cell holds its block and no stray is left '
                 '(server placement and destroy records); exact gifts; the anomaly monitor switches the building head and '
                 'the dig head; a session rule presses the hotbar key of the needed block or the pickaxe (rule_select, not '
                 'policy); %d neutral steps after success' % HOLD_AFTER_STEPS)
    limit_s = limit_for(mode in BEACON_MODES, mode in BEACON_REPAIR_MODES)
    meta = dict(arm=arm, checkpoint=None if checkpoint is None else str(Path(checkpoint).resolve()), use_stage=bool(use_stage),
                mode=mode, limit_s=limit_s, frame_every=FRAME_EVERY, teacher_check=bool(teacher_check), scope=scope)
    if teacher != 'round1': meta['teacher'] = teacher          # round 1 records stay as they were
    if dig_checkpoint is not None: meta['dig_checkpoint'] = str(Path(dig_checkpoint).resolve())
    if attack_decode != 'sample': meta['attack_decode'] = attack_decode
    for task in tasks:
        config, fixture, plan = prepare(batch, arm, root, task, meta['checkpoint'], use_stage, teacher_check, teacher,
                                        occluded=mode in OCCLUDED_MODES, large=mode in LARGE_MODES,
                                        repair=mode in REPAIR_MODES, beacon=mode in BEACON_MODES, dig=mode in DIG_MODES,
                                        beacon_repair=mode in BEACON_REPAIR_MODES, dig_checkpoint=meta.get('dig_checkpoint'),
                                        attack_decode=attack_decode)
        files = [config, fixture, plan, root / REGISTRY, root / 'src/minecraftrl/build_task.py',
                 root / 'src/minecraftrl/place_task.py',
                 root / 'src/minecraftrl/PlaceObservation.java', root / 'src/minecraftrl/world_save_patch.py']
        if teacher != 'round1': files.append(root / 'src/minecraftrl/build_recover.py')
        if teacher in SENSOR_TEACHERS:
            files += [root / 'src/minecraftrl' / name for name in ('build_sensor.py', 'SensorObservation.java',
                      'pipeline/sensor.py', 'pipeline/seen_map.py', 'pipeline/sensor_guard.py', 'pipeline/session.py')]
        if teacher in (SENSOR_DIG, SENSOR_REPAIR):
            files += [root / 'src/minecraftrl' / name for name in ('build_sensor_repair.py', 'head_switch.py',
                                                                    'build_repair.py', 'pipeline/cli.py')]
        if attack_decode != 'sample':
            files += [root / 'src/minecraftrl' / name for name in ('attack_decode.py', 'reacquire_head.py')]
        if mode in LARGE_MODES: files.append(root / 'src/minecraftrl/build_large.py')
        if mode in BEACON_MODES + BEACON_REPAIR_MODES: files.append(root / 'src/minecraftrl/build_beacon.py')
        if mode in REPAIR_MODES:
            files += [root / 'src/minecraftrl/build_repair.py', root / 'src/minecraftrl/pipeline/session.py']
        if not teacher_check: files.append(root / registry['embedding'])
        if checkpoint: files.append(Path(checkpoint))
        if dig_checkpoint: files.append(Path(dig_checkpoint))
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
                           '--limit-s', str(limit_s), '--frame-every', str(FRAME_EVERY)]
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
                            **{k: stage[k] for k in ('strays_left', 'dug', 'broken', 'unknown_destroys', 'wrong_left',
                                                     'repairs', 'collected', 'collect_lost', 'anomalies', 'unaccounted')
                               if k in stage},
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
    p.add_argument('--dig-checkpoint', type=Path, help='the dig head next to the building head (beacon_repair)')
    p.add_argument('--attack-decode', choices=ATTACK_DECODES, default='sample',
                   help="'threshold': the dig head's attack key from its own probability (comparison arm, 2026-11-24)")
    args = p.parse_args(argv)
    if not args.teacher_check and args.checkpoint is None: p.error('--checkpoint is required for a policy arm')
    checkpoint = None if args.checkpoint in (None, 'official') else Path(args.checkpoint)
    return collect(batch=args.batch, arm=args.arm, checkpoint=checkpoint, use_stage=args.use_stage == 'on', mode=args.mode,
                   tasks=args.tasks, seeds=args.seeds, sampling=args.sampling, teacher_check=args.teacher_check,
                   teacher=args.teacher, dig_checkpoint=args.dig_checkpoint, attack_decode=args.attack_decode)


if __name__ == '__main__':
    sys.exit(main())
