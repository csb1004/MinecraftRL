"""Iron demo entry point: a supervisor owns one game worker, STOP, and the deadline.

probe   runs a fixed step list (preparation and skill checks; no API calls).
_worker is the game process launched by the supervisor.
"""
import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from ..runtime import StopGate, write_json
from .steps import Step

STARTUP_LIMIT_S = 600.
CLEANUP_GRACE_S = 60.
EMBEDDING_ROOTS = ('weights', 'results')


def load_skills(root, specs, names):
    """Load only the requested condition embeddings; each must be a hashed, bounded project file."""
    import numpy as np
    root = Path(root).resolve(); skills = {}
    for name in sorted(set(names)):
        spec = specs[name]
        path = (root / spec['embedding']).resolve()
        if not any(path.is_relative_to(root / part) for part in EMBEDDING_ROOTS) or path.stat().st_size > 16384:
            raise ValueError('embedding must be a bounded file under weights/ or results/: ' + name)
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != spec['sha256']: raise ValueError('embedding hash mismatch: ' + name)
        value = np.load(io.BytesIO(blob), allow_pickle=False)
        if value.shape != (1, 512) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid embedding: ' + name)
        scale = spec['cond_scale']
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 <= scale <= 8:
            raise ValueError('invalid cond_scale: ' + name)
        skills[name] = dict(spec, embedding=value, cond_scale=float(scale), file=spec['embedding'])
    return skills


def runtime_needs(steps, config, *, oversight):
    """(policy skills to load, whether the recipe catalog is needed). With oversight a replan may use any selected
    skill and craft steps, so load them before the game starts."""
    skills = {s.skill for s in steps if s.kind == 'policy'}
    if oversight: skills |= {v for v in (config.get('selected') or {}).values() if v}
    return sorted(skills), oversight or any(s.kind == 'rule' for s in steps)


def load_plan(path):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    steps = [Step(id=v['id'], skill=v['skill'], kind=v['kind'], target=v['target'], count=v['count'],
                  block=v.get('block'), claim_ids=tuple(v.get('claim_ids', ())), params=v.get('params', {}))
             for v in values]
    if len({s.id for s in steps}) != len(steps): raise ValueError('duplicate step id')
    return steps


def parse(argv=None):
    p = argparse.ArgumentParser(prog='minecraftrl.pipeline.cli')
    p.add_argument('mode', choices=('probe', '_worker'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--plan-json', type=Path, help='fixed plan; without it the planner plans from the knowledge store at start')
    p.add_argument('--fixture', type=Path)
    p.add_argument('--stop-file', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--sampling-seed', type=int)
    p.add_argument('--limit-s', type=float)
    p.add_argument('--frame-every', type=int, default=1)
    p.add_argument('--goal', help='ITEM:COUNT turns on goal-derived guard and monitors (oversight)')
    p.add_argument('--kb', type=Path, help='SQLite knowledge store under results/ (default: <output>/kb.sqlite)')
    p.add_argument('--utterance', help='user request such as "철 캐줘": the LLM proposes the goal, the program validates it '
                                       'against config allowed_goals (needs --allow-paid; instead of --goal)')
    p.add_argument('--allow-paid', action='store_true',
                   help='allow paid OpenAI calls in the slow path (OPENAI_API_KEY, OPENAI_MODEL; cap: config budgets.llm_calls)')
    args = p.parse_args(argv)
    args.root = args.root.resolve(); args.output = args.output.resolve()
    if not args.output.is_relative_to(args.root / 'results'): raise ValueError('output must be under results/')
    if args.goal is not None:
        item, _, count = args.goal.partition(':')
        if not item or not count.isdigit() or int(count) < 1: raise ValueError('goal must be ITEM:COUNT with COUNT >= 1')
        args.goal = (item, int(count))
    if args.plan_json is None and args.goal is None and args.utterance is None:
        raise ValueError('without --plan-json a goal (--goal or --utterance) is needed to plan from the knowledge store')
    if args.utterance is not None:
        if not args.utterance.strip(): raise ValueError('utterance must not be empty')
        if not args.allow_paid: raise ValueError('--utterance needs --allow-paid (the goal is proposed by the LLM)')
        if args.goal is not None: raise ValueError('give either --goal or --utterance, not both')
    if args.kb is not None:
        args.kb = args.kb.resolve()
        if not args.kb.is_relative_to(args.root / 'results'): raise ValueError('knowledge store must be under results/')
    return args


def initial_steps(kb, config, goal):
    """(plan, concrete steps) from the knowledge store with the empty start inventory: naive on an empty store, the full
    chain on an inherited one (spec 5.12)."""
    from dataclasses import replace
    from .planner import concrete_skill, plan
    initial = plan(goal[0], goal[1], {}, kb, config['game_context'])
    return initial, [replace(s, skill=concrete_skill(s.skill, config.get('selected') or {})) for s in initial.steps]


def request_goal(llm, utterance, config):
    """The LLM proposes, the program validates (spec 2 scene 1); a count the request omits is the caller's criterion."""
    from .goal import parse_goal
    allowed = {item: tuple(bounds) for item, bounds in config['allowed_goals'].items()}
    return parse_goal(llm, utterance, allowed=allowed, default_count=int(config['success_criterion']['count']))


def settings(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    seed = config['seed'] if args.seed is None else args.seed
    # World seed fixes terrain; the sampling seed varies policy trials in the same world.
    sampling = seed if args.sampling_seed is None else args.sampling_seed
    limit_s = float(config['limit_s'] if args.limit_s is None else args.limit_s)
    if not 0 < limit_s <= 900: raise ValueError('limit_s must be in (0, 900]')
    return config, seed, sampling, limit_s


def worker(args):
    gate = StopGate(); env = policy = recorder = kb = None; started_wall = time.time(); minecraft_dir = None
    signal.signal(signal.SIGTERM, lambda *_: gate.stop('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: gate.stop('SIGINT'))
    output = args.output
    try:
        config, seed, sampling, limit_s = settings(args)
        steps = load_plan(args.plan_json) if args.plan_json else []
        policy_skills, need_catalog = runtime_needs(steps, config, oversight=bool(args.goal or args.utterance))
        skills = load_skills(args.root, config['skills'], policy_skills)
        import numpy as np
        import torch
        random.seed(sampling); np.random.seed(sampling); torch.manual_seed(sampling); torch.cuda.manual_seed_all(sampling)
        sys.path.insert(0, str(args.root / 'external/steve1'))
        from steve1.VPT.agent import ENV_KWARGS
        from functools import partial
        from minerl.env.malmo import InstanceManager
        from minerl.utils import process_watcher
        from ..minerl_runtime import launch_scoped_watcher, prepare_jar_cache
        from ..policies import MineRLEnvironment, Steve1Policy
        from ..workshop_monitor import workshop_survival
        from ..workshop_recipes import RecipeCatalog
        from ..world_save_patch import build_save_jar
        from .recorder import Recorder
        from .session import DemoSession, fixture_sha52
        process_watcher.launch = partial(launch_scoped_watcher, output)
        minecraft_dir = Path(InstanceManager.MINECRAFT_DIR)
        jar = build_save_jar(args.root, minecraft_dir, craft_monitor=True, workshop_monitor=True, demo_monitor=True)
        prepare_jar_cache(minecraft_dir, output, source_override=jar)
        sha52 = None
        if args.fixture:
            fixture = args.fixture.resolve(); sha52 = fixture_sha52(fixture)
            os.environ['MINECRAFTRL_DEMO_FIXTURE'] = str(fixture)
        else: os.environ.pop('MINECRAFTRL_DEMO_FIXTURE', None)
        catalog = RecipeCatalog.from_jar(jar, extra_ids=('stone_pickaxe',)) if need_catalog else None
        env = MineRLEnvironment(workshop_survival(**ENV_KWARGS, terminate_on_death=True, demo=True).make())
        preparation = None
        if policy_skills:
            from ..cuda_startup import prime_convolution
            # Precomputed embeddings only: MineCLIP and the prior are not loaded in the game process.
            policy = Steve1Policy(env.raw, args.root, prompt_profile='official-visual')
            preparation = prime_convolution(policy.agent.policy.net.img_process.cnn.stacks[0].firstconv.layer)
        recorder = Recorder(output, frame_every=args.frame_every)
        oversight = None; llm = None
        if args.allow_paid:
            from .openai_client import ResponsesClient
            llm = ResponsesClient(os.getenv('OPENAI_MODEL'), os.getenv('OPENAI_API_KEY'), allow_paid=True,
                                  max_calls=int(config['budgets']['llm_calls']))
        if args.utterance:
            try: goal = request_goal(llm, args.utterance, config)
            except (RuntimeError, ValueError) as exc:
                recorder.event('goal_rejected', utterance=args.utterance, reason=str(exc))
                write_json(output / 'summary.json', dict(status='goal_rejected', execution_valid=False, steps=0, reason=str(exc),
                                                         utterance=args.utterance, api_calls=len(llm.calls),
                                                         call_log=[{k: v for k, v in c.items() if k != 'usage'} for c in llm.calls]))
                return None
            recorder.event('goal', utterance=args.utterance, **goal); args.goal = (goal['item'], goal['count'])
        if args.goal:
            from .kb import KnowledgeBase
            from .oversight import Oversight
            kb = KnowledgeBase(args.kb or output / 'kb.sqlite')
            if args.plan_json is None:
                initial, steps = initial_steps(kb, config, args.goal)
                recorder.event('initial_plan', plan_id=initial.id, naive=initial.naive, problems=initial.problems,
                               steps=[asdict(x) for x in steps])
                if not steps: raise ValueError('planner produced no steps: %s' % initial.problems)
            oversight = Oversight(kb, config['game_context'], args.goal[0], steps, run_id=output.name, llm=llm,
                                  selected=config.get('selected'), goal_count=args.goal[1])
            oversight.max_retries = int(config['budgets']['step_retries'])
            oversight.max_rounds = int(config['budgets'].get('knowledge_rounds', oversight.max_rounds))
        env.seed(seed)
        session = DemoSession(env, policy, catalog, recorder, gate, args.stop_file, limit_s=limit_s, skills=skills,
                              fixture_sha52=sha52, settle_steps=int(config.get('settle_steps', 20)),
                              tool_lock=bool(config.get('policy_tool_lock', False)), oversight=oversight)
        report = session.run(steps)
        report.update(mode=args.mode, seed=seed, sampling_seed=sampling, limit_s=limit_s, fixture=str(args.fixture) if args.fixture else None,
                      fixture_sha52=sha52, jar=str(jar), jar_receipt=json.loads(jar.with_suffix('.json').read_text()),
                      skills={k: {f: v for f, v in s.items() if f != 'embedding'} for k, s in skills.items()},
                      backend_preparation=preparation, api_calls=report['oversight']['api_calls'] if report.get('oversight') else 0,
                      controller_scope='STEVE-1 policy steps; rule GUI steps and scripted probe steps are recorded separately'
                                       + ('; policy hotbar/drop/inventory keys masked (tool keeping is not policy skill)' if session.tool_lock else ''),
                      gpu=dict(torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))
        write_json(output / 'summary.json', report)
        return 0 if report['execution_valid'] else 2
    except BaseException as exc:
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(output / 'summary.json', dict(status='error', execution_valid=False, steps=0,
                   error=type(exc).__name__ + ': ' + str(exc), phase='loading_or_startup'))
        return 2
    finally:
        if kb is not None: kb.close()
        for resource in (policy, env):
            if resource is not None:
                try: resource.close()
                except BaseException as exc: (output / 'close-error.txt').write_text(repr(exc), encoding='utf-8')
        if recorder is not None: recorder.close()
        if minecraft_dir is not None:
            try:
                log = minecraft_dir / 'logs/latest.log'
                if log.is_file() and log.stat().st_mtime >= started_wall: shutil.copyfile(log, output / 'minecraft-latest.log')
            except OSError as exc: (output / 'minecraft-log-copy-error.txt').write_text(repr(exc))


def session_start(output):
    """Monotonic time of the worker's session_started event, or None."""
    path = output / 'timeline.jsonl'
    if not path.exists(): return None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try: event = json.loads(line)
            except json.JSONDecodeError: return None
            if event.get('kind') == 'session_started': return float(event['t'])
    return None


def supervise(args):
    from ..processes import ProcessOwner
    config, seed, sampling, limit_s = settings(args)
    if args.plan_json: load_plan(args.plan_json)
    args.output.mkdir(parents=True, exist_ok=False)
    stop = args.stop_file or args.output / 'STOP'
    write_json(args.output / 'request.json', dict(mode=args.mode, seed=seed, sampling_seed=sampling, limit_s=limit_s, config=str(args.config),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        plan=json.loads(args.plan_json.read_text(encoding='utf-8')) if args.plan_json else 'planner_from_kb_at_start', fixture=str(args.fixture) if args.fixture else None,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest() if args.fixture else None,
        stop_file=str(stop), frame_every=args.frame_every, api_calls_allowed=bool(args.allow_paid),
        openai_model=os.getenv('OPENAI_MODEL') if args.allow_paid else None,
        goal=list(args.goal) if args.goal else None, utterance=args.utterance, kb=str(args.kb) if args.kb else None,
        kb_sha256_before=hashlib.sha256(args.kb.read_bytes()).hexdigest() if args.kb and args.kb.exists() else None))
    argv = [sys.executable, '-m', 'minecraftrl.pipeline.cli', '_worker', '--root', str(args.root), '--output', str(args.output),
            '--config', str(args.config), '--stop-file', str(stop),
            '--seed', str(seed), '--sampling-seed', str(sampling), '--limit-s', str(limit_s), '--frame-every', str(args.frame_every)]
    if args.plan_json: argv.extend(['--plan-json', str(args.plan_json)])
    if args.fixture: argv.extend(['--fixture', str(args.fixture)])
    if args.goal: argv.extend(['--goal', '%s:%d' % args.goal])
    if args.utterance: argv.extend(['--utterance', args.utterance])
    if args.kb: argv.extend(['--kb', str(args.kb)])
    if args.allow_paid: argv.append('--allow-paid')
    owner = ProcessOwner(); force_reason = None; started = time.monotonic()
    with (args.output / 'console.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(argv, cwd=str(args.root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            stop_seen = None
            while proc.poll() is None:
                owner.capture(); now = time.monotonic()
                active = session_start(args.output)
                deadline = started + STARTUP_LIMIT_S if active is None else active + limit_s + CLEANUP_GRACE_S
                if stop.exists() and stop_seen is None: stop_seen = now; force_reason = 'cancelled'
                if stop_seen is not None and now - stop_seen >= CLEANUP_GRACE_S: break
                if now >= deadline:
                    force_reason = force_reason or ('startup_timeout' if active is None else 'hard_timeout'); break
                time.sleep(.05)
        except KeyboardInterrupt:
            force_reason = 'cancelled'; stop.touch(exist_ok=True)
        finally:
            cleanup = owner.close(proc)
    path = args.output / 'summary.json'
    report = json.loads(path.read_text()) if path.exists() else dict(status=force_reason or 'worker_failed',
        execution_valid=False, error=force_reason or 'worker exit ' + str(proc.returncode), shutdown_summary_missing=True)
    if force_reason == 'cancelled' and report.get('status') != 'stop_file':
        report['shutdown_worker_status'] = report.get('status'); report['status'] = 'cancelled'
    report['process_cleanup'] = cleanup; report['worker_exit_code'] = proc.returncode
    if cleanup['remaining_live_pids']: report['status'] = 'worker_failed'; report['execution_valid'] = False
    write_json(path, report)
    return report


def main(argv=None):
    args = parse(argv)
    if args.mode == '_worker': return worker(args)
    report = supervise(args)
    print(json.dumps({k: report.get(k) for k in ('status', 'execution_valid', 'all_steps_completed', 'steps')}))
    return 0 if report.get('execution_valid') else 2


if __name__ == '__main__':
    sys.exit(main())
