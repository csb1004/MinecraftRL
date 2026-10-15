import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback
from .preflight import inspect, command
from .runtime import run_episode, StopGate, write_json
from .reporting import aggregate
from .processes import ProcessOwner


def sampling_seed(value):
    value=int(value)
    if not 0<=value<2**32:raise argparse.ArgumentTypeError('sampling seed must be in [0, 2**32)')
    return value


def resolve_rl_sampling_seed(args):
    value=getattr(args,'rl_sampling_seed',None)
    return args.seed if value is None else value


def parser():
    p = argparse.ArgumentParser(description='MinecraftRL official policy execution probe')
    p.add_argument('operation', choices=['preflight', 'episode', 'suite', 'session', 'craft-probe', '_worker'])
    p.add_argument('--craft-probe',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--craft-stop-on-success',action='store_true',help='Stop experimental crafting at verified inventory recovery')
    p.add_argument('--craft-recover-cursor',action='store_true',help='Experimental read-only GUI monitor and explicit cursor-return cleanup')
    p.add_argument('--craft-no-quick-move',action='store_true',help='Experimental explicit sneak/Shift action constraint in craft-probe')
    p.add_argument('--craft-result-slot-guard',action='store_true',help='Experimental Shift constraint at predicted crafting result slot')
    p.add_argument('--craft-workshop',action='store_true',help='Explicit multi-recipe workstation GUI experiment')
    p.add_argument('--workshop-goals',nargs='+',choices=['wooden_pickaxe','wooden_axe'],default=['wooden_pickaxe','wooden_axe'])
    p.add_argument('--craft-recipe-book',action='store_true',help='Explicit recipe-book GUI experiment without learned policy')
    p.add_argument('--rl-collect',action='store_true')
    p.add_argument('--rl-checkpoint',type=Path)
    p.add_argument('--rl-embed',type=Path)
    p.add_argument('--rl-waypoint-actor',type=Path,help='Development-only learned waypoint head with known fixture goals')
    p.add_argument('--rl-navigation-trial',type=Path,help='Explicit bounded navigation correction/evaluation/RL protocol')
    p.add_argument('--rl-waypoint-start',choices=['stall','initial'],default='stall',help='Explicit initial-state diagnostic or normal stall handoff')
    p.add_argument('--rl-waypoint-planner',choices=['fixture','visit_memory','adaptive_rule','adaptive_llm'],default='fixture',help='Development route planner; adaptive_llm needs a separately authorized Windows broker')
    p.add_argument('--rl-navigation-strategy',choices=['visited','coverage','coverage_motion'],default='visited',help='RM candidate ranking: destination visits or observed path coverage with optional motion feedback; no terrain oracle')
    p.add_argument('--rl-recovery-embed',type=Path,help='Eval-only single graph-stall goal-conditioning intervention')
    planner=p.add_mutually_exclusive_group()
    planner.add_argument('--rl-planner-choice',choices=['explore_then_resume','remain_paused'],help='Local fixed planner fixture; no API calls')
    planner.add_argument('--rl-planner-exchange',type=Path,help='One-use Windows broker exchange under project results')
    planner.add_argument('--rl-rm-plan-fixture',choices=['explore','unsupported','subgoals'],help='Fixed subgoal DAG fixture; no LLM')
    p.add_argument('--rl-skill-catalogue',type=Path,help='Opt-in verified neural subgoal embedding catalogue')
    p.add_argument('--rl-rm-navigation-actor',type=Path,help='Experimental waypoint executor inside the RM subgoal planner')
    p.add_argument('--rl-rm-plan',action='store_true',help='Opt-in subgoal DAG protocol for the planner mailbox')
    p.add_argument('--rl-planner-max-requests',type=int,choices=[1,2,3],default=1)
    p.add_argument('--rl-planner-exchange-live',action='store_true',help='Accept separately authorized live advisory scope')
    p.add_argument('--rl-planner-delay',type=float,default=5.,help='Local fixed response delay in seconds')
    p.add_argument('--rl-split',choices=['train','eval'],default='train')
    p.add_argument('--rl-world',type=Path)
    p.add_argument('--rl-sampling-seed',type=sampling_seed,help='Policy RNG only; defaults to world seed')
    p.add_argument('--rl-prepare',action='store_true')
    p.add_argument('--rl-probe-route',choices=['break','detour','death'])
    p.add_argument('--rl-teacher',action='store_true',help='Explicit privileged correction demonstration on training fixtures')
    p.add_argument('--rl-student-correction',action='store_true',help='Explicit student stall -> neutral -> privileged teacher demonstration')
    p.add_argument('--rl-reward-mode',choices=['sparse','wood-potential-v1','wood-potential-v2','wood-potential-v3','wood-potential-v4'],default='sparse')
    p.add_argument('--task-session', action='store_true')
    p.add_argument('--task-priorities',action='store_true',help='Persistent multiple collection goals with aging and prerequisite priority inheritance')
    p.add_argument('--world-slot', help='Persistent local world slot (session only; default companion)')
    p.add_argument('--world-store-root',type=Path,help='Project-local isolated world store for experiments')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--policy', required=True, choices=['vpt', 'steve1'])
    p.add_argument('--condition', choices=['spontaneous', 'wood', 'dirt', 'planks', 'sticks', 'table'])
    p.add_argument('--seed', type=int, default=104729)
    p.add_argument('--limit-s', type=float, default=180)
    p.add_argument('--no-progress-s', type=float, default=60, help='Task-session inventory stall threshold, seconds (0,180]')
    p.add_argument('--steve-prompt-profile', choices=['short','official'],default='short')
    p.add_argument('--output', type=Path)
    p.add_argument('--stop-file', type=Path)
    return p


def condition_for(root, policy, name):
    if policy=='steve1' and name in ('planks','sticks','table'):
        from .crafting_probe import RECIPES
        recipe=RECIPES[name]
        return dict(policy='steve1',name=name,goal=recipe['goal'],target_items=[recipe['target']])
    conditions = json.loads((root/'configs/experiment.json').read_text())['conditions']
    candidates = [c for c in conditions if c['policy'] == policy and (not name or c['name'] == name)]
    if len(candidates) != 1: raise ValueError('select a matching --condition (STEVE-1: wood or dirt)')
    return candidates[0]


def worker(args, condition):
    output = args.output; env = policy = None; minecraft_dir = None; world_store=None
    worker_started_wall = time.time()
    gate = StopGate()
    signal.signal(signal.SIGTERM, lambda *_: gate.stop('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: gate.stop('SIGINT'))
    try:
        import numpy as np
        import torch
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(args.root/'external'/args.policy))
        if args.policy == 'vpt':
            from agent import ENV_KWARGS
        else:
            from steve1.VPT.agent import ENV_KWARGS
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        from minerl.env.malmo import InstanceManager
        from functools import partial
        from minerl.utils import process_watcher
        from .minerl_runtime import launch_scoped_watcher
        process_watcher.launch=partial(launch_scoped_watcher,output)
        minecraft_dir = Path(InstanceManager.MINECRAFT_DIR)
        from .minerl_runtime import prepare_jar_cache
        jar=None
        if getattr(args,'world_slot',None):
            from .world_save_patch import build_save_jar
            jar=build_save_jar(args.root,minecraft_dir,craft_monitor=getattr(args,'craft_recover_cursor',False),workshop_monitor=getattr(args,'craft_workshop',False))
            from .minerl_runtime import configure_player_identity
            player_uuid=configure_player_identity(minecraft_dir,args.world_slot)
            write_json(output/'player-identity.json',{'slot':args.world_slot,'requested_uuid':player_uuid,
                                                    'launcher_username':'MinecraftRL'})
        if args.rl_collect:
            from .world_save_patch import build_save_jar
            from .workshop_monitor import workshop_survival
            jar=build_save_jar(args.root,minecraft_dir,craft_monitor=True,workshop_monitor=True)
            HumanSurvival=workshop_survival
            from .minerl_runtime import configure_player_identity
            identity=configure_player_identity(minecraft_dir,'rl-seed-'+str(args.seed))
            write_json(output/'player-identity.json',dict(uuid=identity,seed=args.seed))
        prepare_jar_cache(InstanceManager.MINECRAFT_DIR, output,source_override=jar)
        from .policies import VPTPolicy, Steve1Policy, MineRLEnvironment
        restore_bundle=None
        if getattr(args,'world_slot',None):
            import uuid
            from .world_store import WorldStore
            from .policies import PersistentMineRLEnvironment
            world_store=WorldStore(args.world_store_root or args.root/'data/worlds',args.world_slot); world_store.__enter__()
            restore_bundle=world_store.load()
            temporary=args.root/'data/worlds/runtime'/uuid.uuid4().hex; temporary.mkdir(parents=True)
            # Pin the loader's temporary extraction inside the actual project.
            os.environ['JAVA_TOOL_OPTIONS']=os.environ.get('JAVA_TOOL_OPTIONS','')+' -Djava.io.tmpdir="'+str(temporary)+'"'
            survival=HumanSurvival
            if getattr(args,'craft_recover_cursor',False):
                from .craft_gui import monitored_survival
                survival=monitored_survival
            if args.craft_workshop:
                from .workshop_monitor import workshop_survival
                survival=workshop_survival
            raw=survival(**ENV_KWARGS,load_filename=restore_bundle['archive'] if restore_bundle else None).make()
            env=PersistentMineRLEnvironment(raw,minecraft_dir,temporary)
        elif args.rl_collect:
            from .rl_worlds import checked_world
            import uuid
            from .policies import PersistentMineRLEnvironment
            temporary=args.root/'data/worlds/runtime'/uuid.uuid4().hex;temporary.mkdir(parents=True)
            os.environ['JAVA_TOOL_OPTIONS']=os.environ.get('JAVA_TOOL_OPTIONS','')+' -Djava.io.tmpdir="'+str(temporary)+'"'
            world=checked_world(args.rl_world,args.seed) if args.rl_world else None
            if args.rl_teacher or args.rl_student_correction:
                from .correction_teacher import validate_teacher_world
                validate_teacher_world(world or {},args.rl_split)
                if args.rl_probe_route or args.rl_prepare or (args.rl_teacher and args.rl_checkpoint):
                    raise ValueError('teacher cannot combine probe, preparation, or learned checkpoint')
            raw=HumanSurvival(**ENV_KWARGS,load_filename=str(args.rl_world) if args.rl_world else None,
                              terminate_on_death=True).make()
            env=PersistentMineRLEnvironment(raw,minecraft_dir,temporary)
        else:
            env = MineRLEnvironment(HumanSurvival(**ENV_KWARGS).make())
        if args.rl_collect and args.rl_prepare:
            from .rl_worlds import prepare_world
            report=prepare_world(env,output,args.seed,gate,args.stop_file)
            return 0 if report['execution_valid'] else 2
        if args.rl_collect and args.rl_teacher:
            from .correction_teacher import CorrectionTeacher
            policy=CorrectionTeacher(env.raw,world)
        elif args.rl_collect and args.rl_probe_route:
            if args.rl_split!='eval':raise ValueError('scripted probes cannot train')
            from .fixture_probe import ProbePolicy
            policy=ProbePolicy(env.raw,world or {},args.rl_probe_route)
        elif args.rl_collect:
            from .rl_policy import RLPolicy
            policy=RLPolicy(env.raw,args.root,output,args.rl_checkpoint,args.rl_embed,resolve_rl_sampling_seed(args),reward_mode=args.rl_reward_mode)
            if getattr(args,'rl_navigation_trial',None):
                from .navigation_trial import NavigationTrial
                trial=json.loads(args.rl_navigation_trial.read_text(encoding='utf-8'))
                if trial['world_sha256']!=world['sha256']:raise ValueError('navigation trial world mismatch')
                expected_split='train' if args.rl_split=='train' else 'development'
                if trial['split']!=expected_split:raise ValueError('navigation trial split mismatch')
                policy=NavigationTrial(policy,env.raw,world,output,trial,args.root)
            elif getattr(args,'rl_waypoint_actor',None):
                from .waypoint_control import WaypointController
                if args.rl_waypoint_planner.startswith('adaptive_'):
                    from .adaptive_control import AdaptiveWaypointController as WaypointController
                policy=WaypointController(policy,args.rl_waypoint_actor,world,output,start_mode=args.rl_waypoint_start,
                    planner_mode=args.rl_waypoint_planner)
            elif getattr(args,'rl_recovery_embed',None):
                if getattr(args,'rl_rm_plan',False) or getattr(args,'rl_rm_plan_fixture',None):
                    from .rm_planning_control import RMPlanningController,FixedRMPlanProvider
                    from .recovery_exchange import FileRMPlanProvider
                    provider=(FileRMPlanProvider(args.rl_planner_exchange,live=args.rl_planner_exchange_live,
                                               max_calls=args.rl_planner_max_requests) if args.rl_planner_exchange else
                              FixedRMPlanProvider(args.rl_planner_delay,unsupported=args.rl_rm_plan_fixture=='unsupported',
                                                  max_calls=args.rl_planner_max_requests))
                    controller=RMPlanningController;extra={}
                    if getattr(args,'rl_skill_catalogue',None):
                        from .subgoal_control import NeuralSubgoalController,FixedSubgoalProvider
                        from .subgoal_skills import load_catalogue
                        controller=NeuralSubgoalController
                        extra['catalogue']=load_catalogue(args.rl_skill_catalogue,args.root)
                        if args.rl_rm_plan_fixture=='subgoals':
                            provider=FixedSubgoalProvider(args.rl_planner_delay,max_calls=args.rl_planner_max_requests)
                        if getattr(args,'rl_rm_navigation_actor',None):
                            from .rm_navigation import RMNavigationController,FixedNavigationProvider
                            if not world or world.get('experimental_split')!='development':
                                raise ValueError('RM navigation integration requires a development world')
                            controller=RMNavigationController;extra['navigation_checkpoint']=args.rl_rm_navigation_actor
                            extra['navigation_strategy']=getattr(args,'rl_navigation_strategy','visited')
                            if args.rl_rm_plan_fixture=='subgoals':
                                provider=FixedNavigationProvider(args.rl_planner_delay,max_calls=args.rl_planner_max_requests)
                    policy=controller(policy,args.rl_recovery_embed,output,raw=env.raw,
                        world_id=world['sha256'],provider=provider,max_requests=args.rl_planner_max_requests,**extra)
                elif getattr(args,'rl_planner_choice',None) or getattr(args,'rl_planner_exchange',None):
                    from .planning_recovery import PlanningRecoveryController,FixedRecoveryProvider
                    from .recovery_exchange import FileRecoveryProvider
                    provider=(FileRecoveryProvider(args.rl_planner_exchange,live=args.rl_planner_exchange_live)
                              if args.rl_planner_exchange else FixedRecoveryProvider(args.rl_planner_choice,args.rl_planner_delay))
                    policy=PlanningRecoveryController(policy,args.rl_recovery_embed,output,raw=env.raw,
                        world_id=world['sha256'],provider=provider)
                else:
                    from .goal_recovery import GoalRecoveryController
                    policy=GoalRecoveryController(policy,args.rl_recovery_embed,output)
            elif args.rl_student_correction:
                from .student_correction import StudentCorrection
                policy=StudentCorrection(policy,env.raw,world,output)
            elif policy.has_trained_tail and args.rl_split!='eval':
                raise ValueError('tail checkpoint PPO collection is not supported; do not drop tail weights')
            if gate.requested.is_set() or (args.stop_file and args.stop_file.exists()):
                raise InterruptedError('stop before RL environment reset')
        elif args.craft_workshop:
            from .workshop import WorkshopController
            from .workshop_recipes import RecipeCatalog
            policy=WorkshopController(RecipeCatalog.from_jar(jar),restore_bundle['observation']['inventory'],args.workshop_goals,output)
        elif args.craft_recipe_book:
            from .recipe_book import RecipeBookController
            policy=RecipeBookController(args.condition,output)
        else: policy = (VPTPolicy(env.raw,args.root) if args.policy=='vpt' else
                  Steve1Policy(env.raw,args.root,prompt_profile=args.steve_prompt_profile))
        def gpu():
            return {'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                    'torch_peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                    'scope': 'PyTorch process including model loading, not total device VRAM'}
        if args.rl_collect and getattr(args,'rl_navigation_trial',None):
            from .navigation_trial import run_navigation_trial
            report=run_navigation_trial(policy,env,output,world=world,seed=args.seed,limit_s=args.limit_s,
                gate=gate,stop_file=args.stop_file,synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and getattr(args,'rl_waypoint_actor',None):
            from .waypoint_control import run_waypoint
            report=run_waypoint(policy,env,output,world=world,seed=args.seed,limit_s=args.limit_s,
                gate=gate,stop_file=args.stop_file,synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and getattr(args,'rl_recovery_embed',None):
            from .goal_recovery import run_goal_recovery
            report=run_goal_recovery(policy,env,output,world=world,split=args.rl_split,
                seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and args.rl_student_correction:
            from .student_correction import run_student_correction
            report=run_student_correction(policy,env,output,world=world,split=args.rl_split,
                seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and args.rl_teacher:
            from .correction_teacher import run_teacher
            report=run_teacher(policy,env,output,world=world,seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and args.rl_probe_route:
            from .fixture_probe import run_probe
            report=run_probe(policy,env,output,world=world,seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect:
            from .rl_collect import run_rl
            report=run_rl(policy,env,output,split=args.rl_split,world=world,reward_mode=args.rl_reward_mode,seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.craft_workshop:
            from .workshop import run_workshop
            report=run_workshop(policy,env,output,bundle=restore_bundle,store=world_store,
                seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif getattr(args,'craft_probe',False):
            from .crafting_probe import run_crafting_probe
            report=run_crafting_probe(policy,env,output,bundle=restore_bundle,store=world_store,
                stop_on_success=args.craft_stop_on_success,
                recover_cursor=args.craft_recover_cursor,
                recipe=args.condition,
                no_quick_move=args.craft_no_quick_move,
                result_slot_guard=args.craft_result_slot_guard,
                seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.task_session:
            from .task_runtime import run_task_session
            from .scheduled_tasks import ScheduledTasks
            report = run_task_session(policy,env,output,seed=args.seed,limit_s=args.limit_s,
                                     no_progress_s=args.no_progress_s,
                                     gate=gate,stop_file=args.stop_file,
                                     synchronize=torch.cuda.synchronize,gpu_metrics=gpu,
                                     world_store=world_store,restore_bundle=restore_bundle,
                                     scheduled_tasks=ScheduledTasks() if getattr(args,'task_priorities',False) else None)
        else:
            report = run_episode(policy, env, output, seed=args.seed, goal=condition['goal'],
                                 target_items=condition['target_items'], limit_s=args.limit_s,
                                 gate=gate, stop_file=args.stop_file,
                                 synchronize=torch.cuda.synchronize, gpu_metrics=gpu)
        return 0 if report['execution_valid'] else 2
    except BaseException as exc:
        (output/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        cleanup = {}
        for label, resource in [('policy', policy), ('environment', env)]:
            if resource:
                try:
                    if label == 'environment': resource.release()
                    resource.close(); cleanup[label] = 'closed'
                except BaseException as close_error: cleanup[label] = repr(close_error)
        write_json(output/'summary.json', {'status':'error','execution_valid':False,'steps':0,
                   'error':repr(exc),'phase':'loading_or_startup','cleanup':cleanup})
        return 2
    finally:
        if world_store is not None and world_store.lock is not None:
            world_store.__exit__(None,None,None)
        if minecraft_dir is not None:
            try:
                log = minecraft_dir/'logs/latest.log'
                if log.is_file() and log.stat().st_mtime >= worker_started_wall:
                    shutil.copyfile(log, output/'minecraft-latest.log')
            except OSError as exc:
                (output/'minecraft-log-copy-error.txt').write_text(repr(exc))



def episode(args, condition):
    task_session=getattr(args,'task_session',False)
    prompt_profile=getattr(args,'steve_prompt_profile','short')
    world_store_root=getattr(args,'world_store_root',None)
    if task_session and args.policy!='steve1': raise ValueError('task sessions require STEVE-1')
    args.output.mkdir(parents=True, exist_ok=False)
    readiness = inspect(args.root, args.policy); write_json(args.output/'preflight.json', readiness)
    write_json(args.output/'request.json', {'policy':args.policy,'condition':condition,'seed':args.seed,
                                          'rl_sampling_seed':resolve_rl_sampling_seed(args) if getattr(args,'rl_collect',False) else None,
                                          'limit_s':args.limit_s,'controller':'workshop_gui' if getattr(args,'craft_workshop',False) else 'recipe_book_gui' if getattr(args,'craft_recipe_book',False) else 'policy','task_session':task_session,
                                          'learned_policy_loaded':not (getattr(args,'craft_recipe_book',False) or getattr(args,'craft_workshop',False) or getattr(args,'rl_prepare',False) or getattr(args,'rl_teacher',False) or getattr(args,'rl_probe_route',None)),
                                          'steve_prompt_profile':prompt_profile,
                                          'world_store_root':str(world_store_root) if world_store_root else None})
    if readiness['status'] == 'blocked':
        report = {'status':'blocked','execution_valid':False,'steps':0,'problems':readiness['problems']}
        write_json(args.output/'summary.json', report); return report
    write_json(args.output/'pip-freeze.json', command([sys.executable, '-m', 'pip', 'freeze', '--all']))
    stop = args.stop_file or args.output/'STOP'
    argv = [sys.executable,'-m','minecraftrl.cli','_worker','--root',str(args.root),'--policy',args.policy,
            '--condition',condition['name'],'--seed',str(args.seed),'--limit-s',str(args.limit_s),
            '--output',str(args.output),'--stop-file',str(stop)]
    if task_session: argv.extend(['--task-session','--no-progress-s',str(getattr(args,'no_progress_s',60.))])
    if getattr(args,'task_priorities',False): argv.append('--task-priorities')
    if getattr(args,'craft_probe',False): argv.append('--craft-probe')
    if getattr(args,'craft_stop_on_success',False): argv.append('--craft-stop-on-success')
    if getattr(args,'craft_recover_cursor',False): argv.append('--craft-recover-cursor')
    if getattr(args,'craft_no_quick_move',False): argv.append('--craft-no-quick-move')
    if getattr(args,'craft_result_slot_guard',False): argv.append('--craft-result-slot-guard')
    if getattr(args,'craft_recipe_book',False): argv.append('--craft-recipe-book')
    if getattr(args,'craft_workshop',False): argv.extend(['--craft-workshop','--workshop-goals',*args.workshop_goals])
    if getattr(args,'rl_collect',False):
        argv.extend(['--rl-collect','--rl-split',args.rl_split])
        argv.extend(['--rl-reward-mode',getattr(args,'rl_reward_mode','sparse')])
        if getattr(args,'rl_sampling_seed',None) is not None:argv.extend(['--rl-sampling-seed',str(args.rl_sampling_seed)])
        if getattr(args,'rl_probe_route',None):argv.extend(['--rl-probe-route',args.rl_probe_route])
        if getattr(args,'rl_teacher',False):argv.append('--rl-teacher')
        if getattr(args,'rl_student_correction',False):argv.append('--rl-student-correction')
        if args.rl_checkpoint:argv.extend(['--rl-checkpoint',str(args.rl_checkpoint)])
        if args.rl_embed:argv.extend(['--rl-embed',str(args.rl_embed)])
        if getattr(args,'rl_navigation_trial',None):argv.extend(['--rl-navigation-trial',str(args.rl_navigation_trial)])
        if getattr(args,'rl_waypoint_actor',None):argv.extend(['--rl-waypoint-actor',str(args.rl_waypoint_actor),
            '--rl-waypoint-start',args.rl_waypoint_start,'--rl-waypoint-planner',args.rl_waypoint_planner])
        if getattr(args,'rl_recovery_embed',None):argv.extend(['--rl-recovery-embed',str(args.rl_recovery_embed)])
        if getattr(args,'rl_planner_choice',None):
            argv.extend(['--rl-planner-choice',args.rl_planner_choice,'--rl-planner-delay',str(args.rl_planner_delay)])
        if getattr(args,'rl_rm_plan_fixture',None):
            argv.extend(['--rl-rm-plan-fixture',args.rl_rm_plan_fixture,'--rl-planner-delay',str(args.rl_planner_delay)])
        if getattr(args,'rl_rm_plan',False):argv.append('--rl-rm-plan')
        if getattr(args,'rl_skill_catalogue',None):argv.extend(['--rl-skill-catalogue',str(args.rl_skill_catalogue)])
        if getattr(args,'rl_rm_navigation_actor',None):argv.extend(['--rl-rm-navigation-actor',str(args.rl_rm_navigation_actor),
            '--rl-navigation-strategy',getattr(args,'rl_navigation_strategy','visited')])
        argv.extend(['--rl-planner-max-requests',str(getattr(args,'rl_planner_max_requests',1))])
        if getattr(args,'rl_planner_exchange',None):
            argv.extend(['--rl-planner-exchange',str(args.rl_planner_exchange)])
            if args.rl_planner_exchange_live:argv.append('--rl-planner-exchange-live')
        if args.rl_world:argv.extend(['--rl-world',str(args.rl_world)])
        if args.rl_prepare:argv.append('--rl-prepare')
    argv.extend(['--steve-prompt-profile',prompt_profile])
    if getattr(args,'world_slot',None): argv.extend(['--world-slot',args.world_slot])
    if world_store_root: argv.extend(['--world-store-root',str(world_store_root)])
    config = json.loads((args.root/'configs/experiment.json').read_text())
    rl_finalization_grace=float(config.get('rl_finalization_grace_s',60.))
    if not 0<rl_finalization_grace<=120: raise ValueError('invalid RL finalization grace')
    startup_deadline = time.monotonic() + config['startup_limit_s']
    active_deadline = None; stop_requested_at = None; force_reason = None
    finalization_deadline = None; cancelled_archive=False; archive_timed_out=False
    samples = []; gpu_error = None; next_sample = 0.
    owner = ProcessOwner()
    cleanup = None
    with (args.output/'console.log').open('w', encoding='utf-8') as log:
        supervisor_worker_started=time.monotonic()
        proc = subprocess.Popen(argv, cwd=str(args.root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                owner.capture()
                now = time.monotonic()
                if active_deadline is None:
                    events = args.output/'events.jsonl'
                    if events.exists():
                        with events.open(encoding='utf-8') as stream:
                            first = stream.readline()
                        if first:
                            event = json.loads(first)
                            if event.get('event') == 'episode_started': active_deadline = event['monotonic_s'] + args.limit_s
                deadline = active_deadline if active_deadline is not None else startup_deadline
                if getattr(args,'rl_collect',False) and finalization_deadline is None:
                    try:
                        marker=json.loads((args.output/'rl-control-finished.json').read_text())
                        stamp=float(marker['monotonic_s'])
                        if marker['control_closed'] is True and marker['release']=='neutral_command_sent' and math.isfinite(stamp) and supervisor_worker_started<=stamp<=now:
                            finalization_deadline=stamp+rl_finalization_grace
                            cancelled_archive=marker.get('cancelled_archive') is True
                    except (OSError,ValueError,KeyError,TypeError): pass
                if stop.exists() and force_reason!='cancelled':
                    stop_requested_at=now;force_reason='cancelled'
                if finalization_deadline is not None and (cancelled_archive or (not stop.exists() and force_reason!='cancelled')):
                    if now>=finalization_deadline:
                        archive_timed_out=True
                        if force_reason!='cancelled':force_reason='rl_finalize_timeout'
                        break
                elif stop.exists() or now >= deadline:
                    if stop_requested_at is None:
                        stop_requested_at = now
                        force_reason = 'cancelled' if stop.exists() else ('time_limit' if active_deadline else 'startup_timeout')
                        # Automatic deadlines never latch the user's shared STOP.
                    if now - stop_requested_at >= config['shutdown_grace_s']:
                        break
                if now >= next_sample:
                    try:
                        smi = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'], capture_output=True, text=True, timeout=.5)
                        if smi.returncode == 0: samples.append({'monotonic_s':now,'device_used_mib':float(smi.stdout.strip())})
                        else: gpu_error = smi.stderr.strip()
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc: gpu_error = repr(exc)
                    next_sample = now + .25
                time.sleep(.02)
        except KeyboardInterrupt:
            force_reason = 'cancelled'; stop.touch(exist_ok=True)
        finally:
            # MineRL starts Java with setsid; a worker group alone is insufficient.
            cleanup = owner.close(proc)
    path = args.output/'summary.json'
    if path.exists(): report = json.loads(path.read_text())
    else:
        report = {'status':('cancelled' if force_reason == 'cancelled' else 'hard_timeout') if force_reason else 'worker_failed','execution_valid':False,
                  'error':force_reason or ('worker exit '+str(proc.returncode)), 'release':'unverified; owned process tree shutdown attempted',
                  'shutdown_summary_missing':True}
        # A neutral step can finish before a slow close exceeds the shutdown
        # grace. Preserve flushed evidence without calling the forced run valid.
        events = args.output/'events.jsonl'
        if events.exists():
            shutdown_acknowledged=False
            with events.open(encoding='utf-8') as stream:
                for line in stream:
                    try: event = json.loads(line)
                    except json.JSONDecodeError: continue  # killed during final write
                    if event.get('event') == 'stop_acknowledged': shutdown_acknowledged=True
                    if shutdown_acknowledged and event.get('event') == 'input_release' and event.get('controller') == 'cleanup':
                        report['release'] = event['outcome']
                        report['release_evidence'] = 'events.jsonl:input_release'
    if force_reason == 'cancelled':
        report['shutdown_worker_status'] = report['status']
        report['status'] = 'cancelled'; report['execution_valid'] = False
    report['process_cleanup'] = cleanup
    if cleanup['remaining_live_pids']:
        report['status'] = 'worker_failed'; report['execution_valid'] = False
    report['worker_exit_code'] = proc.returncode
    if proc.returncode != 0:
        report['execution_valid'] = False
        planned_pause=(report.get('status')=='planning_paused' and proc.returncode==2
            and report.get('release')=='neutral_command_sent'
            and not any(k in report for k in ('error','environment_close_error','policy_close_error')))
        if not planned_pause and report.get('status') not in ('error', 'cancelled', 'hard_timeout', 'worker_failed'): report['status'] = 'worker_failed'
    if getattr(args,'rl_collect',False):
        report['rl_finalization_deadline']=finalization_deadline
        report['rl_cancelled_archive']=cancelled_archive
        report['rl_archive_timed_out']=archive_timed_out
        if not report['execution_valid']: report['rl_training_eligible']=False
    report['device_vram'] = {'sampled_peak_used_mib':max((x['device_used_mib'] for x in samples),default=None),
        'sample_count':len(samples),'sampling_error':gpu_error,'scope':'entire device 0 including other processes; sampled, may miss short peaks'}
    write_json(args.output/'device-vram-samples.json', samples); write_json(path, report)
    return report


def validate_goal_recovery_args(args):
    path=getattr(args,'rl_recovery_embed',None)
    exchange=getattr(args,'rl_planner_exchange',None)
    rm_fixture=getattr(args,'rl_rm_plan_fixture',None)
    rm_plan=getattr(args,'rl_rm_plan',False) or bool(rm_fixture)
    catalogue=getattr(args,'rl_skill_catalogue',None)
    navigation_actor=getattr(args,'rl_rm_navigation_actor',None)
    if navigation_actor:
        if (not catalogue or not rm_plan or getattr(args,'rl_waypoint_actor',None) or
                rm_fixture not in (None,'subgoals')):
            raise SystemExit('RM navigation requires a subgoal catalogue and RM plan, without standalone waypoint control')
        if not navigation_actor.resolve().is_relative_to(args.root.resolve()/'results'):
            raise SystemExit('RM navigation weights must be project-local results')
    if catalogue and not rm_plan:raise SystemExit('Neural catalogue requires the RM planner')
    if rm_fixture=='subgoals' and not catalogue:raise SystemExit('Subgoal fixture requires a neural catalogue')
    if catalogue:
        from .subgoal_skills import load_catalogue
        load_catalogue(catalogue,args.root)
    if rm_plan and (path is None or getattr(args,'rl_planner_choice',None) or not (exchange or rm_fixture)):
        raise SystemExit('RM planner requires explicit recovery and a fixture or mailbox')
    if getattr(args,'rl_planner_max_requests',1)!=1 and not rm_plan:
        raise SystemExit('multiple requests require the RM plan protocol')
    if getattr(args,'rl_planner_exchange_live',False) and exchange is None:
        raise SystemExit('live planner scope requires a mailbox')
    if exchange is not None:
        if path is None or getattr(args,'rl_planner_choice',None):
            raise SystemExit('mailbox requires explicit eval goal recovery and no fixed planner')
        if not exchange.resolve().is_relative_to(args.root.resolve()/'results'):
            raise SystemExit('planner mailbox must be project-local results')
        if exchange.exists():raise SystemExit('planner mailbox must be a new one-use directory')
    if getattr(args,'rl_planner_choice',None) or rm_fixture:
        if path is None:raise SystemExit('planner fixture requires explicit eval goal recovery')
        if not math.isfinite(args.rl_planner_delay) or not 0<=args.rl_planner_delay<=60:
            raise SystemExit('planner fixture delay must be finite and in [0,60]')
    if path is None:return
    if (not args.rl_collect or args.rl_split!='eval' or not args.rl_world or not args.rl_checkpoint or not args.rl_embed
            or args.rl_teacher or args.rl_student_correction or args.rl_probe_route or args.rl_prepare
            or args.rl_reward_mode!='wood-potential-v4'):
        raise SystemExit('goal recovery requires eval, explicit world/checkpoint/wood embedding and v4; no other controller')
    if not path.resolve().is_relative_to(args.root.resolve()/'results'):
        raise SystemExit('goal recovery embedding must be project-local results')


def validate_waypoint_args(args):
    if getattr(args,'rl_navigation_strategy','visited')!='visited' and not getattr(args,'rl_rm_navigation_actor',None):
        raise SystemExit('coverage exploration requires RM navigation actor')
    path=getattr(args,'rl_waypoint_actor',None)
    if path is None:
        if getattr(args,'rl_waypoint_start','stall')!='stall':raise SystemExit('initial waypoint start requires waypoint actor')
        if getattr(args,'rl_waypoint_planner','fixture')!='fixture':raise SystemExit('exploration planner requires waypoint actor')
        return
    if (not args.rl_collect or args.rl_split!='eval' or not args.rl_world or not args.rl_checkpoint or not args.rl_embed
            or args.rl_teacher or args.rl_student_correction or args.rl_probe_route or args.rl_prepare
            or args.rl_recovery_embed or args.rl_reward_mode!='wood-potential-v4'):
        raise SystemExit('waypoint diagnostic requires isolated eval parent/world/embedding and v4')
    if not path.resolve().is_relative_to(args.root.resolve()/'results'):
        raise SystemExit('waypoint weights must be project-local results')


def validate_navigation_trial_args(args):
    path=getattr(args,'rl_navigation_trial',None)
    if not path:return
    if (not args.rl_collect or not args.rl_world or not args.rl_checkpoint or not args.rl_embed or
            args.rl_reward_mode!='wood-potential-v4' or args.rl_prepare or args.rl_probe_route or
            args.rl_teacher or args.rl_student_correction or args.rl_waypoint_actor or args.rl_recovery_embed or
            args.rl_rm_navigation_actor):raise SystemExit('navigation trial requires isolated parent/world/embedding and v4')
    if not path.resolve().is_relative_to(args.root.resolve()/'results'):raise SystemExit('navigation protocol must be project-local')


def main():
    args = parser().parse_args(); args.root = args.root.resolve()
    validate_goal_recovery_args(args)
    validate_waypoint_args(args)
    validate_navigation_trial_args(args)
    if args.rl_student_correction and (not args.rl_collect or not args.rl_world or not args.rl_checkpoint or not args.rl_embed
            or args.rl_teacher or args.rl_probe_route or args.rl_prepare or args.rl_reward_mode!='wood-potential-v4'):
        raise SystemExit('student correction requires an explicit checkpoint, embedding, reserved world and v4; no other controller')
    if args.rl_sampling_seed is not None and not args.rl_collect:raise SystemExit('--rl-sampling-seed requires RL collection')
    if args.rl_prepare and (not args.rl_collect or args.rl_world):raise SystemExit('prepare requires fresh RL world')
    if args.rl_world:
        if not args.rl_collect or not args.rl_world.resolve().is_relative_to(args.root/'results'):raise SystemExit('RL world must be project-local results archive')
    if args.rl_collect and (args.operation not in ('episode','_worker') or args.policy!='steve1' or args.condition!='wood' or args.world_slot or args.task_session or args.craft_probe):
        raise SystemExit('RL collection requires an isolated fresh STEVE-1 wood episode')
    if args.operation=='craft-probe': args.craft_probe=True; args.condition=args.condition or 'planks'
    if args.craft_stop_on_success and not args.craft_probe: raise SystemExit('--craft-stop-on-success requires craft-probe')
    if args.craft_recover_cursor and not args.craft_probe: raise SystemExit('--craft-recover-cursor requires craft-probe')
    if args.craft_no_quick_move and not (args.craft_probe and args.craft_recover_cursor):
        raise SystemExit('--craft-no-quick-move requires craft-probe with cursor recovery')
    if args.craft_result_slot_guard and not (args.craft_probe and args.craft_recover_cursor):
        raise SystemExit('--craft-result-slot-guard requires craft-probe with cursor recovery')
    if args.craft_no_quick_move and args.craft_result_slot_guard: raise SystemExit('select one craft action guard')
    if args.craft_recipe_book and not (args.craft_probe and args.craft_recover_cursor and args.condition in ('sticks','table')):
        raise SystemExit('--craft-recipe-book requires sticks/table craft-probe with cursor recovery')
    if args.craft_recipe_book and (args.craft_no_quick_move or args.craft_result_slot_guard):
        raise SystemExit('recipe-book experiment uses its own explicit no-Shift GUI controller')
    if args.craft_workshop and (not (args.craft_probe and args.craft_recover_cursor) or args.craft_recipe_book or args.craft_no_quick_move or args.craft_result_slot_guard):
        raise SystemExit('workshop requires a separate craft-probe with cursor recovery')
    if args.craft_probe:
        if args.operation not in ('craft-probe','_worker') or args.condition not in ('planks','sticks','table'):
            raise SystemExit('craft-probe is a separate experimental operation')
        if args.policy!='steve1' or args.task_session or not args.world_slot or not args.world_store_root:
            raise SystemExit('craft-probe requires STEVE-1 and an isolated existing world slot')
        if not args.world_store_root.resolve().is_relative_to(args.root/'results'):
            raise SystemExit('craft-probe world store must be inside project results')
        if args.steve_prompt_profile!='short': raise SystemExit('craft-probe uses one fixed experimental prompt')
    elif args.condition in ('planks','sticks','table'): raise SystemExit('crafting condition requires craft-probe')
    if args.operation=='session':
        if args.policy!='steve1': raise SystemExit('session requires --policy steve1')
        args.task_session=True; args.condition='wood'
        args.world_slot=args.world_slot or 'companion'
    if args.task_priorities and not args.task_session: raise SystemExit('--task-priorities requires a task session')
    if args.world_slot and not (args.task_session or args.craft_probe): raise SystemExit('--world-slot requires a task session or craft-probe')
    if args.world_store_root:
        args.world_store_root=args.world_store_root.resolve()
        if not (args.task_session or args.craft_probe) or not args.world_store_root.is_relative_to(args.root):
            raise SystemExit('--world-store-root requires a task session and a path inside the project')
    if not 0 < args.limit_s <= 180: raise SystemExit('--limit-s must be in (0, 180]')
    if not 0 < args.no_progress_s <= 180: raise SystemExit('--no-progress-s must be in (0,180]')
    if args.output: args.output = args.output.resolve()
    if args.stop_file: args.stop_file = args.stop_file.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.operation == 'preflight':
        args.output = args.output or args.root/'results'/('preflight-'+args.policy+'-'+stamp)
        args.output.mkdir(parents=True,exist_ok=False)
        result = inspect(args.root, args.policy); write_json(args.output/'preflight.json', result)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return 2 if result['problems'] else 0
    if args.operation == 'suite':
        config = json.loads((args.root/'configs/experiment.json').read_text())
        base = args.output or args.root/'results'/('suite-'+args.policy+'-'+stamp)
        selected = [c for c in config['conditions'] if c['policy'] == args.policy and (not args.condition or args.condition == c['name'])]
        if not selected: raise ValueError('no matching policy/condition')
        base.mkdir(parents=True,exist_ok=False)
        summaries = {c['name']: aggregate([], planned=len(config['seeds']), has_target=bool(c['target_items'])) for c in selected}
        for c in selected:
            reports = []
            for seed in config['seeds']:
                args.seed = seed; args.output = base/(c['name']+'-'+str(seed))
                report = episode(args, c); reports.append(report)
                # Feasibility gate: inspect technical failures before consuming
                # the remaining seeds. A valid run without acquisition continues.
                if not report['execution_valid'] or (args.stop_file and args.stop_file.exists()): break
            summaries[c['name']] = aggregate(reports,planned=len(config['seeds']),has_target=bool(c['target_items']))
            if reports and not reports[-1]['execution_valid']: break
        write_json(base/'aggregate.json', summaries)
        print(json.dumps(summaries,indent=2)); return 0 if summaries and all(x['technical_failures']==0 and x['unrun']==0 and x['cancelled']==0 for x in summaries.values()) else 2
    c = condition_for(args.root,args.policy,args.condition)
    args.output = args.output or args.root/'results'/(('session-' if args.task_session else 'episode-')+args.policy+'-'+stamp)
    if args.operation == '_worker': return worker(args,c)
    if args.task_session: print('Task session output: '+str(args.output),flush=True)
    report = episode(args,c); print(json.dumps(report,indent=2))
    # A successfully applied task cancel is a successful control command, not
    # valid gameplay or a completed collection task. Preserve those distinctions.
    clean_task_cancel=(args.task_session and report.get('status')=='cancelled'
                       and (report.get('task') or {}).get('status')=='cancelled'
                       and report.get('release')=='neutral_command_sent'
                       and report.get('worker_exit_code')==2
                       and report.get('process_cleanup',{}).get('remaining_live_pids')==[]
                       and report.get('world_save',{}).get('status')!='failed'
                       and not any(key=='error' or key.endswith('_close_error') for key in report))
    return 0 if report['execution_valid'] or clean_task_cancel else 2

if __name__ == '__main__': raise SystemExit(main())
