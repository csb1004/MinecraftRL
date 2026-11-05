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
from .policies import steve_guidance
from .cli_arguments import parser, sampling_seed


def resolve_rl_sampling_seed(args):
    value=getattr(args,'rl_sampling_seed',None)
    return args.seed if value is None else value


def validate_steve_guidance_args(args):
    guidance=getattr(args,'steve_cond_scale',None)
    if guidance is None:return None
    guidance=steve_guidance(guidance)
    if not (getattr(args,'task_session',False) and args.policy=='steve1') or any(
            getattr(args,k,False) for k in ('rl_collect','craft_probe','craft_recipe_book','craft_workshop')):
        raise ValueError('explicit guidance requires a STEVE-1 task session')
    return guidance


def validate_policy_memory_args(args):
    if getattr(args,'task_policy_memory',False) and not (
            getattr(args,'task_session',False) and args.policy=='steve1' and getattr(args,'world_slot',None)
            and getattr(args,'steve_prompt_profile','short')=='official-visual' and not any(
                getattr(args,k,False) for k in ('rl_collect','craft_probe','craft_recipe_book','craft_workshop'))):
        raise ValueError('policy memory requires an official-visual STEVE-1 persistent task session')




def condition_for(root, policy, name):
    if policy=='steve1' and name in ('tool_cobblestone','tool_axe_wood','tool_workstation_access','tool_recover_axe','tool_evade_hostile'):
        from .tool_collection import PHASES
        spec=PHASES[name.removeprefix('tool_')]
        return dict(policy=policy,name=name,goal=spec['prompt'],target_items=list(spec['items']),target_count=spec['count'])
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
            monitored_recovery=bool(getattr(args,'task_recovery_catalogue',None) or getattr(args,'tool_collection',None))
            jar=build_save_jar(args.root,minecraft_dir,craft_monitor=getattr(args,'craft_recover_cursor',False) or monitored_recovery,workshop_monitor=getattr(args,'craft_workshop',False) or monitored_recovery)
            from .minerl_runtime import configure_player_identity
            player_uuid=configure_player_identity(minecraft_dir,args.world_slot)
            write_json(output/'player-identity.json',{'slot':args.world_slot,'requested_uuid':player_uuid,
                                                    'launcher_username':'MinecraftRL'})
        if args.rl_collect or getattr(args,'rm_conditioning',None):
            from .world_save_patch import build_save_jar
            from .workshop_monitor import workshop_survival
            jar=build_save_jar(args.root,minecraft_dir,craft_monitor=True,workshop_monitor=True,
                startup_control=getattr(args,'rl_controlled_start',False))
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
            if args.craft_workshop or getattr(args,'task_recovery_catalogue',None) or getattr(args,'tool_collection',None):
                from .workshop_monitor import workshop_survival
                survival=workshop_survival
            extra={'terminate_on_death':True} if (getattr(args,'task_recovery_catalogue',None) or getattr(args,'tool_collection',None)) else {}
            raw=survival(**ENV_KWARGS,load_filename=restore_bundle['archive'] if restore_bundle else None,**extra).make()
            env=PersistentMineRLEnvironment(raw,minecraft_dir,temporary)
        elif args.rl_collect:
            from .rl_worlds import checked_world
            import uuid
            from .policies import PersistentMineRLEnvironment
            temporary=args.root/'data/worlds/runtime'/uuid.uuid4().hex;temporary.mkdir(parents=True)
            os.environ['JAVA_TOOL_OPTIONS']=os.environ.get('JAVA_TOOL_OPTIONS','')+' -Djava.io.tmpdir="'+str(temporary)+'"'
            world=checked_world(args.rl_world,args.seed) if args.rl_world else None
            from .startup_control import configure_startup_control
            startup_control=configure_startup_control(args.root,args.rl_world,world,output,
                enabled=getattr(args,'rl_controlled_start',False))
            if args.rl_teacher or args.rl_student_correction:
                from .correction_teacher import validate_teacher_world
                validate_teacher_world(world or {},args.rl_split)
                if args.rl_probe_route or args.rl_prepare or (args.rl_teacher and args.rl_checkpoint):
                    raise ValueError('teacher cannot combine probe, preparation, or learned checkpoint')
            raw=HumanSurvival(**ENV_KWARGS,load_filename=str(args.rl_world) if args.rl_world else None,
                              terminate_on_death=True).make()
            env=PersistentMineRLEnvironment(raw,minecraft_dir,temporary)
        elif getattr(args,'recovery_world',None) or getattr(args,'recovery_prepare',False):
            from .rl_worlds import checked_world
            from .policies import PersistentMineRLEnvironment
            import uuid
            temporary=args.root/'data/worlds/runtime'/uuid.uuid4().hex;temporary.mkdir(parents=True)
            os.environ['JAVA_TOOL_OPTIONS']=os.environ.get('JAVA_TOOL_OPTIONS','')+' -Djava.io.tmpdir="'+str(temporary)+'"'
            world=checked_world(args.recovery_world,args.seed) if args.recovery_world else None
            raw=HumanSurvival(**ENV_KWARGS,load_filename=str(args.recovery_world) if args.recovery_world else None,
                              terminate_on_death=True).make()
            env=PersistentMineRLEnvironment(raw,minecraft_dir,temporary)
            env.recovery_world=world
        else:
            extra={'terminate_on_death':True} if getattr(args,'rm_conditioning',None) else {}
            env = MineRLEnvironment(HumanSurvival(**ENV_KWARGS,**extra).make())
        if args.rl_collect:env.startup_control=startup_control
        if (args.rl_collect and args.rl_prepare) or getattr(args,'recovery_prepare',False):
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
            policy=WorkshopController(None if args.workshop_equip_only else RecipeCatalog.from_jar(jar),
                restore_bundle['observation']['inventory'],args.workshop_goals,output,equip=args.workshop_equip,
                equip_only=args.workshop_equip_only,equip_slot=args.workshop_equip_slot,consume_food=args.workshop_consume_food)
        elif getattr(args,'rm_conditioning',None):
            from .rm_conditioning import ConditioningPolicy
            policy_extra={}
            if args.rm_conditioning in ('official','purpose-fixed','purpose-switch'):
                from .rm_purpose_conditioning import PurposeConditioningPolicy
                ConditioningPolicy = PurposeConditioningPolicy
            elif args.rm_conditioning in ('logo-staged','logo-fixed'):
                from .rm_logo_conditioning import LogoConditioningPolicy
                ConditioningPolicy = LogoConditioningPolicy
            elif args.rm_conditioning=='goal-queue':
                from .rm_program import ProgramPolicy
                ConditioningPolicy=ProgramPolicy
                policy_extra['program']=args.rm_program
                if getattr(args,'s1_record',False):
                    from .s1_learning import LearningPolicy
                    ConditioningPolicy=LearningPolicy
                    policy_extra.update(checkpoint=args.s1_checkpoint,split=args.s1_split)
                if getattr(args,'s2_variant',None):
                    from .s2_learning import SemanticPolicy
                    ConditioningPolicy=SemanticPolicy
                    policy_extra.update(variant=args.s2_variant,checkpoint=args.s2_checkpoint,
                                        split=args.s2_split,action_budget=args.s2_action_budget)
                if getattr(args,'recovery_variant',None):
                    from .recovery_learning import RecoveryPolicy
                    ConditioningPolicy=RecoveryPolicy
                    if getattr(args,'effect_learning',False):
                        from .effect_learning import EffectPolicy
                        ConditioningPolicy=EffectPolicy
                    if getattr(args,'execution_learning',False):
                        from .execution_learning import ExecutionPolicy
                        ConditioningPolicy=ExecutionPolicy
                    policy_extra.update(variant=args.recovery_variant,checkpoint=args.recovery_checkpoint,
                        split=args.recovery_split,action_budget=args.recovery_action_budget,task=args.recovery_task)
            policy_seed=args.seed if getattr(args,'recovery_sampling_seed',None) is None else args.recovery_sampling_seed
            policy=ConditioningPolicy(env.raw,args.root,output,args.rm_conditioning,args.rm_goal_bank,policy_seed,
                lambda:gate.requested.is_set() or bool(args.stop_file and args.stop_file.exists()),**policy_extra)
        elif args.craft_recipe_book:
            from .recipe_book import RecipeBookController
            policy=RecipeBookController(args.condition,output)
        else: policy = (VPTPolicy(env.raw,args.root) if args.policy=='vpt' else
                  Steve1Policy(env.raw,args.root,prompt_profile=args.steve_prompt_profile,
                               guidance=getattr(args,'steve_cond_scale',None)))
        def gpu():
            return {'torch_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                    'torch_peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                    'scope': 'PyTorch process including model loading, not total device VRAM'}
        if getattr(args,'rm_conditioning',None):
            from .rm_conditioning import run_conditioning
            report=run_conditioning(policy,env,output,seed=args.seed,limit_s=args.limit_s,
                gate=gate,stop_file=args.stop_file,synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.rl_collect and getattr(args,'rl_navigation_trial',None):
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
        elif args.tool_collection:
            from .tool_collection import run_tool_collection
            report=run_tool_collection(policy,env,output,bundle=restore_bundle,store=world_store,phase=args.tool_collection,
                condition_path=args.tool_condition,health_profile=args.tool_health_profile,
                seed=args.seed,limit_s=args.limit_s,gate=gate,stop_file=args.stop_file,
                synchronize=torch.cuda.synchronize,gpu_metrics=gpu)
        elif args.task_session:
            from .task_runtime import run_task_session
            from .scheduled_tasks import ScheduledTasks
            from .cuda_startup import prime_convolution
            # Load CUDA convolution kernels before reset and the control clock.
            # This invokes no recurrent state, sampling, or environment actions.
            write_json(output/'backend-preparation-started.json',dict(phase='task_session_policy_initialization'))
            preparation=prime_convolution(policy.agent.policy.net.img_process.cnn.stacks[0].firstconv.layer)
            write_json(output/'backend-preparation.json',dict(enabled=True,**preparation))
            policy_memory=None
            if getattr(args,'task_policy_memory',False):
                from .policy_memory import PolicyMemory,memory_identity,initialized_template
                policy_memory=PolicyMemory(policy,memory_identity(policy,args.root),initialized_template(policy),restore_bundle)
            recovery=None
            if getattr(args,'task_recovery_catalogue',None):
                from .session_recovery import SessionRecovery
                from .subgoal_skills import load_catalogue
                skill=load_catalogue(args.task_recovery_catalogue,args.root)['leave_stalled_area']
                np.save(output/'recovery-embed.npy',skill['embedding'])
                hazard_skill=None
                if getattr(args,'task_hazard_condition',None):
                    from .hazard_condition import load_condition
                    hazard_skill=load_condition(args.task_hazard_condition,args.root)
                    np.save(output/'hazard-embed.npy',hazard_skill['embedding'])
                recovery=SessionRecovery(policy,skill,(restore_bundle or {}).get('controller_state',{}).get('neural_recovery'),
                    threat_guard=getattr(args,'task_threat_guard',False),hazard_skill=hazard_skill,
                    pursuit_guard=getattr(args,'task_pursuit_guard',False))
            report = run_task_session(policy,env,output,seed=args.seed,limit_s=args.limit_s,
                                     no_progress_s=args.no_progress_s,
                                     gate=gate,stop_file=args.stop_file,
                                     synchronize=torch.cuda.synchronize,gpu_metrics=gpu,
                                     world_store=world_store,restore_bundle=restore_bundle,
                                     allow_startup_pickups=getattr(args,'allow_startup_pickups',False),
                                     recovery=recovery,
                                     policy_memory=policy_memory,
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
    validate_policy_memory_args(args)
    guidance=validate_steve_guidance_args(args)
    prompt_profile=getattr(args,'steve_prompt_profile','short')
    world_store_root=getattr(args,'world_store_root',None)
    if task_session and args.policy!='steve1': raise ValueError('task sessions require STEVE-1')
    if getattr(args,'allow_startup_pickups',False) and not (task_session and getattr(args,'world_slot',None)):
        raise ValueError('Startup pickup accounting requires a persistent task session')
    if getattr(args,'task_threat_guard',False) and not getattr(args,'task_recovery_catalogue',None):
        raise ValueError('Threat guard requires neural session recovery')
    if getattr(args,'task_pursuit_guard',False) and not getattr(args,'task_threat_guard',False):
        raise ValueError('Pursuit guard requires threat guard')
    if getattr(args,'task_hazard_condition',None):
        if not getattr(args,'task_threat_guard',False):raise ValueError('Hazard condition requires threat guard')
        from .hazard_condition import load_condition
        load_condition(args.task_hazard_condition,args.root)
    if getattr(args,'task_recovery_catalogue',None):
        if not (task_session and getattr(args,'world_slot',None)):
            raise ValueError('Neural session recovery requires a persistent task session')
        from .subgoal_skills import load_catalogue
        load_catalogue(args.task_recovery_catalogue,args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    readiness = inspect(args.root, args.policy); write_json(args.output/'preflight.json', readiness)
    write_json(args.output/'request.json', {'policy':args.policy,'condition':condition,'seed':args.seed,
                                          'rl_sampling_seed':resolve_rl_sampling_seed(args) if getattr(args,'rl_collect',False) else None,
                                          'limit_s':args.limit_s,'controller':'workshop_gui' if getattr(args,'craft_workshop',False) else 'recipe_book_gui' if getattr(args,'craft_recipe_book',False) else 'policy','task_session':task_session,
                                          'learned_policy_loaded':not (getattr(args,'craft_recipe_book',False) or getattr(args,'craft_workshop',False) or getattr(args,'rl_prepare',False) or getattr(args,'rl_teacher',False) or getattr(args,'rl_probe_route',None)),
                                          'steve_prompt_profile':prompt_profile,
                                          'steve_cond_scale':guidance,
                                          'task_policy_memory':getattr(args,'task_policy_memory',False),
                                          'world_store_root':str(world_store_root) if world_store_root else None,
                                          'allow_startup_pickups':getattr(args,'allow_startup_pickups',False),
                                          'task_recovery_catalogue':str(args.task_recovery_catalogue) if getattr(args,'task_recovery_catalogue',None) else None,
                                          'task_threat_guard':getattr(args,'task_threat_guard',False),
                                          'task_pursuit_guard':getattr(args,'task_pursuit_guard',False),
                                          'task_hazard_condition':str(args.task_hazard_condition) if getattr(args,'task_hazard_condition',None) else None,
                                          'recovery_source':'local_rule_privileged_server' if getattr(args,'task_recovery_catalogue',None) else None})
    if readiness['status'] == 'blocked':
        report = {'status':'blocked','execution_valid':False,'steps':0,'problems':readiness['problems']}
        write_json(args.output/'summary.json', report); return report
    write_json(args.output/'pip-freeze.json', command([sys.executable, '-m', 'pip', 'freeze', '--all']))
    stop = args.stop_file or args.output/'STOP'
    argv = [sys.executable,'-m','minecraftrl.cli','_worker','--root',str(args.root),'--policy',args.policy,
            '--condition',condition['name'],'--seed',str(args.seed),'--limit-s',str(args.limit_s),
            '--output',str(args.output),'--stop-file',str(stop)]
    if getattr(args,'rm_conditioning',None):
        argv.extend(['--rm-conditioning',args.rm_conditioning,'--rm-goal-bank',str(args.rm_goal_bank)])
    if getattr(args,'rm_program',None):argv.extend(['--rm-program',str(args.rm_program)])
    if getattr(args,'s1_record',False):argv.extend(['--s1-record','--s1-split',args.s1_split])
    if getattr(args,'s1_checkpoint',None):argv.extend(['--s1-checkpoint',str(args.s1_checkpoint)])
    if getattr(args,'s2_variant',None):
        argv.extend(['--s2-variant',args.s2_variant,'--s2-split',args.s2_split,'--s2-action-budget',str(args.s2_action_budget)])
    if getattr(args,'s2_checkpoint',None):argv.extend(['--s2-checkpoint',str(args.s2_checkpoint)])
    if getattr(args,'recovery_variant',None):
        argv.extend(['--recovery-variant',args.recovery_variant,'--recovery-task',args.recovery_task,
            '--recovery-split',args.recovery_split,'--recovery-action-budget',str(args.recovery_action_budget)])
    for name in ('recovery_checkpoint','recovery_world','recovery_sampling_seed'):
        if getattr(args,name,None) is not None:argv.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
    if getattr(args,'recovery_prepare',False):argv.append('--recovery-prepare')
    if getattr(args,'effect_learning',False):argv.append('--effect-learning')
    if getattr(args,'execution_learning',False):argv.append('--execution-learning')
    if task_session: argv.extend(['--task-session','--no-progress-s',str(getattr(args,'no_progress_s',60.))])
    if getattr(args,'allow_startup_pickups',False): argv.append('--allow-startup-pickups')
    if getattr(args,'task_recovery_catalogue',None):argv.extend(['--task-recovery-catalogue',str(args.task_recovery_catalogue)])
    if getattr(args,'task_threat_guard',False):argv.append('--task-threat-guard')
    if getattr(args,'task_pursuit_guard',False):argv.append('--task-pursuit-guard')
    if getattr(args,'task_hazard_condition',None):argv.extend(['--task-hazard-condition',str(args.task_hazard_condition)])
    if getattr(args,'task_priorities',False): argv.append('--task-priorities')
    if getattr(args,'task_policy_memory',False):argv.append('--task-policy-memory')
    if getattr(args,'craft_probe',False): argv.append('--craft-probe')
    if getattr(args,'craft_stop_on_success',False): argv.append('--craft-stop-on-success')
    if getattr(args,'craft_recover_cursor',False): argv.append('--craft-recover-cursor')
    if getattr(args,'craft_no_quick_move',False): argv.append('--craft-no-quick-move')
    if getattr(args,'craft_result_slot_guard',False): argv.append('--craft-result-slot-guard')
    if getattr(args,'craft_recipe_book',False): argv.append('--craft-recipe-book')
    if getattr(args,'craft_workshop',False): argv.extend(['--craft-workshop','--workshop-goals',*args.workshop_goals])
    if getattr(args,'workshop_equip',None): argv.extend(['--workshop-equip',args.workshop_equip])
    if getattr(args,'workshop_equip_only',False):argv.append('--workshop-equip-only')
    if getattr(args,'workshop_equip_slot',None) is not None:argv.extend(['--workshop-equip-slot',str(args.workshop_equip_slot)])
    if getattr(args,'workshop_consume_food',None):argv.extend(['--workshop-consume-food',args.workshop_consume_food])
    if getattr(args,'tool_collection',None): argv.extend(['--tool-collection',args.tool_collection])
    if getattr(args,'tool_condition',None): argv.extend(['--tool-condition',str(args.tool_condition)])
    if getattr(args,'tool_health_profile','standard')!='standard': argv.extend(['--tool-health-profile',args.tool_health_profile])
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
        if getattr(args,'rl_controlled_start',False):argv.append('--rl-controlled-start')
        if args.rl_prepare:argv.append('--rl-prepare')
    argv.extend(['--steve-prompt-profile',prompt_profile])
    if guidance is not None:argv.extend(['--steve-cond-scale',str(guidance)])
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
                if (getattr(args,'rl_collect',False) or getattr(args,'tool_collection',None) or getattr(args,'rm_conditioning',None)) and finalization_deadline is None:
                    try:
                        marker_name=('rm-control-finished.json' if getattr(args,'rm_conditioning',None) else
                                     'tool-control-finished.json' if getattr(args,'tool_collection',None) else 'rl-control-finished.json')
                        marker=json.loads((args.output/marker_name).read_text())
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
                        if force_reason!='cancelled':
                            force_reason=('rm_finalize_timeout' if getattr(args,'rm_conditioning',None) else
                                          'tool_finalize_timeout' if getattr(args,'tool_collection',None) else 'rl_finalize_timeout')
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
    if getattr(args,'tool_collection',None):
        report['tool_finalization_deadline']=finalization_deadline
        report['tool_archive_timed_out']=archive_timed_out
    if getattr(args,'rm_conditioning',None):
        report['rm_finalization_deadline']=finalization_deadline
        report['rm_archive_timed_out']=archive_timed_out
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
    from .navigation_refinement import navigation_action_selection
    navigation_action_selection(json.loads(path.read_text(encoding='utf-8-sig')))


def validate_controlled_start_args(args):
    if not getattr(args,'rl_controlled_start',False):return
    if (not args.rl_collect or not args.rl_world or not args.rl_navigation_trial or
            args.rl_prepare or args.rl_teacher or args.rl_student_correction or args.rl_probe_route):
        raise SystemExit('controlled startup currently requires an explicit navigation evaluation')
    trial=json.loads(args.rl_navigation_trial.read_text(encoding='utf-8-sig'))
    if trial.get('mode')!='evaluate' or trial.get('split') not in ('train','development'):
        raise SystemExit('controlled startup is restricted to train/development evaluation')


def main():
    args = parser().parse_args(); args.root = args.root.resolve()
    # The old custom checkpoints, worlds and dedicated workflows were retired by
    # the user. Reject their CLI before preflight, environment startup or loading.
    # Shared collection/math utilities remain available for the new RL design.
    if any(token == '--rl-collect' or token.startswith('--rl-') for token in sys.argv[1:]):
        raise SystemExit('Legacy RL experiment paths were removed on 2026-10-25. Use the official STEVE-1 baseline; the new RL learning loop is not implemented yet.')
    from .rm_conditioning import validate_args as validate_rm_conditioning_args
    validate_rm_conditioning_args(args)
    if args.effect_learning:
        from .goal_contract import read_json
        if (args.rm_conditioning!='goal-queue' or args.recovery_variant!='nonlinear-on'
                or args.recovery_prepare or args.rm_program is None
                or read_json(args.rm_program).get('schema')!='goal_spec_v1'):
            raise ValueError('E1 requires canonical saved-start nonlinear-on recovery controls')
    if args.execution_learning:
        from .goal_contract import read_json
        if (args.effect_learning or args.rm_conditioning!='goal-queue' or args.recovery_variant!='nonlinear-on'
                or args.recovery_prepare or args.rm_program is None
                or read_json(args.rm_program).get('schema')!='goal_spec_v2'):
            raise ValueError('E2 requires v2 saved-start nonlinear-on controls and excludes E1')
    if (any(getattr(args,k,None) for k in ('recovery_variant','recovery_prepare','recovery_world','recovery_checkpoint'))
            or args.recovery_sampling_seed is not None or args.recovery_task!='collection'
            or args.recovery_split!='development' or args.recovery_action_budget!=1200):
        from .recovery_learning import validate_args as validate_recovery_args
        validate_recovery_args(args)
    validate_goal_recovery_args(args)
    validate_waypoint_args(args)
    validate_navigation_trial_args(args)
    validate_controlled_start_args(args)
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
    if args.workshop_equip_only and (not args.craft_workshop or not args.workshop_equip or args.workshop_goals!=[args.workshop_equip]):
        raise SystemExit('equipment-only assistance requires a single explicit workshop tool')
    if args.workshop_equip_slot is not None and (not args.craft_workshop or not args.workshop_equip):
        raise SystemExit('tool slot relocation requires explicit workshop equipment')
    if args.workshop_consume_food and not args.workshop_equip_only:
        raise SystemExit('food assistance requires explicit equipment-only mode')
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
    validate_steve_guidance_args(args)
    validate_policy_memory_args(args)
    if args.task_priorities and not args.task_session: raise SystemExit('--task-priorities requires a task session')
    if args.allow_startup_pickups and not (args.task_session and args.world_slot):
        raise SystemExit('--allow-startup-pickups requires a persistent task session')
    if args.task_recovery_catalogue and not (args.task_session and args.world_slot):
        raise SystemExit('--task-recovery-catalogue requires a persistent task session')
    if args.task_threat_guard and not args.task_recovery_catalogue:
        raise SystemExit('--task-threat-guard requires --task-recovery-catalogue')
    if args.task_hazard_condition and not args.task_threat_guard:
        raise SystemExit('--task-hazard-condition requires --task-threat-guard')
    if args.task_pursuit_guard and not args.task_threat_guard:
        raise SystemExit('--task-pursuit-guard requires --task-threat-guard')
    if args.tool_collection:
        if (args.operation not in ('episode','_worker') or args.policy!='steve1' or args.seed!=104761
                or args.steve_prompt_profile!='official-visual' or not args.world_slot or not args.world_store_root
                or args.task_session or args.craft_probe or args.rl_collect):
            raise SystemExit('tool collection requires a separate official-visual STEVE-1 development episode on seed104761')
        if args.condition!='tool_'+args.tool_collection:raise SystemExit('tool phase and condition must match')
        from .tool_collection import validate_condition_phase,health_limits_for_phase
        try:
            validate_condition_phase(args.tool_collection,args.tool_condition)
            health_limits_for_phase(args.tool_collection,args.tool_health_profile)
        except ValueError as exc:raise SystemExit(str(exc)) from exc
        if not args.world_store_root.resolve().is_relative_to(args.root/'results'):
            raise SystemExit('tool collection world store must be inside project results')
    elif (args.condition in ('tool_cobblestone','tool_axe_wood','tool_workstation_access','tool_recover_axe','tool_evade_hostile')
            or args.tool_condition or args.tool_health_profile!='standard'):
        raise SystemExit('tool condition requires explicit tool collection mode')
    if args.world_slot and not (args.task_session or args.craft_probe or args.tool_collection): raise SystemExit('--world-slot requires a persistent experiment')
    if args.world_store_root:
        args.world_store_root=args.world_store_root.resolve()
        if not (args.task_session or args.craft_probe or args.tool_collection) or not args.world_store_root.is_relative_to(args.root):
            raise SystemExit('--world-store-root requires a task session and a path inside the project')
    if not 0 < args.limit_s <= 180: raise SystemExit('--limit-s must be in (0, 180]')
    if getattr(args,'s2_variant',None):
        if args.rm_conditioning!='goal-queue' or args.rm_program is None or args.s1_record or args.s1_checkpoint:
            raise SystemExit('S2 requires goal-queue/program and cannot mix with S1')
        if not 1 <= args.s2_action_budget <= 1200: raise SystemExit('S2 action budget must be 1 to 1200')
    elif getattr(args,'s2_checkpoint',None): raise SystemExit('S2 checkpoint requires an explicit variant')
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
