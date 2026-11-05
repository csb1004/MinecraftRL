"""One world, one revisable collection task, asynchronous structured controls."""
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
import uuid
from .runtime import StopGate, distribution, inventory_counts, jsonable, save_png, write_json
from .tasks import CommandBus, TaskState, validate_command
from .dialogue import decide
from .memory import atomic_json


def write_command(folder, command):
    """Publish one complete JSON file; readers never see a partially written command."""
    validate_command(command)
    return _publish(folder,command)


def server_inventory_observation(observation,info):
    """Monitored sessions use server counts, retaining client counts as evidence."""
    server=info.get('workshop',{})
    if server.get('server_available')!=1 or server.get('error',0)!=0:
        raise ValueError('authoritative_inventory_unavailable')
    counts=inventory_counts({'inventory':server.get('server_inventory')})
    return dict(observation,client_inventory=inventory_counts(observation),inventory=counts)


def write_dialogue(folder, message):
    """Publish an utterance; the inbox logs its decision before any command."""
    decide(message)
    return _publish(folder,message)


def _publish(folder, command):
    folder=Path(folder); folder.mkdir(exist_ok=True)
    name='%020d-%s'%(time.time_ns(),uuid.uuid4().hex)
    temporary=folder/(name+'.tmp'); final=folder/(name+'.json')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(command,stream,ensure_ascii=False,allow_nan=False)
    temporary.rename(final)
    return final


class DirectoryInbox:
    def __init__(self, folder, bus, emit):
        self.folder=Path(folder); self.folder.mkdir(exist_ok=True)
        self.bus=bus; self.emit=emit; self.finished=threading.Event(); self.error=None
        self.thread=threading.Thread(target=self.read,daemon=True)
    def start(self): self.thread.start()
    def read(self):
        seen=set(); messages={}
        try:
            while not self.finished.wait(.01):
                for path in sorted(self.folder.glob('*.json')):
                    if path.name in seen: continue
                    seen.add(path.name)
                    try:
                        if path.is_symlink() or path.stat().st_size>65536: raise ValueError('invalid command file')
                        command=json.loads(path.read_text(encoding='utf-8'))
                        if isinstance(command,dict) and command.get('kind')=='dialogue':
                            message=command
                            decision=decide(message)
                            mid=message['message_id']
                            if mid in messages:
                                if messages[mid]!=message: raise ValueError('message_id reused with different content')
                                self.emit('dialogue_duplicate',message_id=mid,file=path.name)
                                continue
                            messages[mid]=message
                            self.emit('dialogue_decision',message_id=mid,message=message,decision=decision,file=path.name)
                            command=decision['command']
                            if command is None: continue
                        with self.bus.lock:
                            accepted=self.bus.submit(command)
                            self.emit('command_queued' if accepted else 'command_duplicate',
                                      command_id=command['command_id'],command=command,file=path.name)
                    except (ValueError,OSError) as exc:
                        self.emit('command_rejected',file=path.name,error=str(exc))
        except BaseException as exc:
            self.error=repr(exc)
    def close(self):
        self.finished.set(); self.thread.join(timeout=2)
        if self.thread.is_alive(): raise RuntimeError('command reader did not stop')


class ProgressWatchdog:
    """No observed inventory high-water increase, not a diagnosis of its cause."""
    def __init__(self,threshold_s):
        if type(threshold_s) not in (int,float) or not math.isfinite(threshold_s) or not 0<threshold_s<=180:
            raise ValueError('no_progress_s must be finite and in (0,180]')
        self.threshold_s=threshold_s
        self.reset()

    def reset(self):
        self.key=None; self.best=0; self.since=None

    def sample(self,task,now):
        if not task or task['status']!='running':
            self.reset(); return None
        key=(task['task_id'],task['revision'])
        if key!=self.key or task['progress']>self.best:
            self.key=key; self.best=task['progress']; self.since=now
        elapsed=now-self.since
        return elapsed if elapsed>=self.threshold_s else None


def run_task_session(policy, env, output, *, seed, limit_s=180., stop_file=None,no_progress_s=60.,
                     gate=None, bus=None, save_screen=save_png,
                     synchronize=lambda:None, gpu_metrics=lambda:{},world_store=None,restore_bundle=None,
                     scheduled_tasks=None,allow_startup_pickups=False,recovery=None,policy_memory=None):
    if not 0<limit_s<=180: raise ValueError('limit_s must be in (0,180]')
    watchdog=ProgressWatchdog(no_progress_s)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    frames=output/'frames'; frames.mkdir(exist_ok=True)
    external_bus=bus is not None; bus=bus or CommandBus(); gate=gate or StopGate()
    from .scheduled_tasks import ScheduledTasks
    state=scheduled_tasks if scheduled_tasks is not None else TaskState()
    scheduled=isinstance(state,ScheduledTasks)
    inbox=None; obs=None; sequence=0; started=None; accounted=None
    world_id=uuid.uuid4().hex; saved_task=object()
    world_verified=False
    from copy import deepcopy
    controller_state=deepcopy((restore_bundle or {}).get('controller_state',{}))
    restored_terminal=False
    prepared=None; attempts=0; inference=[]; cycles=[]
    report={'status':'initializing','execution_valid':False,'steps':0,'environment_resets':0,
            'worker_pid':os.getpid(),
            'seed':seed,'controller':'policy','session_kind':'revisable_collection',
            'release':'not_attempted','no_progress_s':no_progress_s,'no_progress_events':0,
            'priority_scheduling':scheduled}
    report['policy_memory']=dict(enabled=policy_memory is not None,
        source_available=bool((restore_bundle or {}).get('policy_memory')))
    log=(output/'events.jsonl').open('w',encoding='utf-8'); log_lock=threading.RLock()
    def emit(event, **fields):
        with log_lock:
            log.write(json.dumps(jsonable(dict(event=event,monotonic_s=time.monotonic(),**fields)),ensure_ascii=False,allow_nan=False)+'\n')
            log.flush()
    def server_terminal(info,phase,raw_done=None):
        server=info.get('workshop',{})
        dead=(recovery is not None and server.get('server_available')==1 and server.get('error',0)==0
              and server.get('server_alive')==0 and server.get('server_health')==0)
        if not dead:return False
        if 'server_death' not in report:
            report['server_death']=dict(phase=phase,raw_done=raw_done,server_tick=server.get('server_tick'),
                server_health=server['server_health'],server_alive=server['server_alive'],
                client_life=obs.get('life_stats'),source='privileged_server')
            emit('server_death_observed',**report['server_death'])
        gate.stop('server_death')
        if report['status'] not in ('error','cancelled'):report['status']='environment_done'
        recovery.end('terminal',time.monotonic(),emit)
        return True
    def halted():
        if stop_file and Path(stop_file).exists(): gate.stop('stop_file')
        if started is not None and time.monotonic()-started>=limit_s: gate.stop('time_limit')
        return gate.requested.is_set()
    def recovery_blocked():
        if recovery is None or not recovery.blocked:return False
        report['status']='recovery_blocked';report['recovery_blocked']=recovery.blocked
        emit('recovery_blocked',reason=recovery.blocked,source='privileged_server')
        recovery.end('recovery_blocked',time.monotonic(),emit)
        gate.stop('recovery_blocked')
        return True
    def checkpoint(force=False):
        nonlocal saved_task
        task=state.checkpoint(world_id) if scheduled else state.snapshot()
        if force or task!=saved_task:
            atomic_json(output/'task-checkpoint.json',state.checkpoint(world_id))
            saved_task=task
    def account(service=False):
        nonlocal accounted
        if not scheduled or accounted is None: return
        now=time.monotonic(); delta=max(0.,now-accounted); accounted=now
        state.advance(delta,service_s=delta if service else 0.)
    def release(reason):
        nonlocal obs,sequence
        outcome=env.release()
        # The neutral step advances MineRL: never infer from the pre-release frame.
        if getattr(env,'last_observation',None) is not None:
            obs=env.last_observation; sequence+=1
            if recovery is not None:obs=server_inventory_observation(obs,env.last_info)
            inventory_counts(obs)
        evidence=getattr(env,'release_evidence',None)
        emit('input_release',controller='cleanup',reason=reason,outcome=outcome,task=state.snapshot(),server_evidence=evidence,
             inventory={k:v for k,v in inventory_counts(obs).items() if v} if obs is not None else None,
             client_inventory={k:v for k,v in obs.get('client_inventory',{}).items() if v} if obs is not None else None)
        if recovery is not None and outcome=='neutral_command_sent' and not (evidence or {}).get('verified'):
            raise ValueError('recovery_input_release_not_verified')
        if recovery is not None:server_terminal(env.last_info,'input_release:'+reason,getattr(env,'terminal',None))
        return outcome
    def observe():
        if sequence<=state.last_observation: return
        observed=state.observe(inventory_counts(obs),sequence)
        events=observed if scheduled else ([observed] if observed else [])
        checkpoint()
        for event in events:
            name=event.pop('event'); emit(name,**event,inventory=inventory_counts(obs))
            if name=='plan_stage_completed' and (not scheduled or event['task']['task_id']==state.current_id):
                release('stage_transition')
    def pause_if_stalled():
        elapsed=(state.stalled_s() if scheduled else watchdog.sample(state.snapshot(),time.monotonic()))
        if scheduled and elapsed<watchdog.threshold_s: return False
        if elapsed is None: return False
        state.pause_for_no_progress(elapsed,watchdog.threshold_s)
        prepared_task=state.snapshot()
        checkpoint()
        release('no_progress')
        emit('task_no_progress',task=prepared_task,observation_sequence=state.last_observation,
             inventory=inventory_counts(obs))
        report['no_progress_events']+=1
        watchdog.reset()
        return True
    try:
        env.seed(seed); obs=env.reset(); report['environment_resets']=1
        if restore_bundle is not None:
            from .world_store import validate_restored_observation, observation_evidence
            write_json(output/'world-restore-observation.json',observation_evidence(obs))
            report['world_restore']=validate_restored_observation(restore_bundle,obs,
                                                                  allow_startup_pickups=allow_startup_pickups)
            world_id=restore_bundle['world_id']
            state=TaskState.restore(restore_bundle['task_checkpoint'],world_id)
            if scheduled and not isinstance(state,ScheduledTasks):
                state=ScheduledTasks.from_legacy(state)
            scheduled=isinstance(state,ScheduledTasks)
            report['priority_scheduling']=scheduled
            restored_terminal=(state.all_terminal() if scheduled else
                               bool(state.task and state.task['status'] in ('completed','cancelled')))
            sequence=state.last_observation+1
        world_verified=True
        if recovery is not None:
            release('recovery_initial_observation')
            report['inventory_source']='privileged_server'
        baseline=inventory_counts(obs); started=time.monotonic(); accounted=started
        checkpoint(force=True)
        emit('episode_started',seed=seed,session_kind='revisable_collection',baseline_inventory=baseline,world_id=world_id,worker_pid=report['worker_pid'])
        if restore_bundle is not None:
            emit('world_restored',task=state.snapshot(),world_id=world_id,evidence=report['world_restore'])
        if not external_bus:
            inbox=DirectoryInbox(output/'commands',bus,emit); inbox.start()
        emit('session_ready',commands=str(output/'commands'))
        while not halted():
            account()
            if inbox and inbox.error: raise RuntimeError('command reader failed: '+inbox.error)
            for command in bus.drain():
                if halted(): break
                try: state.check(command)
                except ValueError as exc:
                    emit('command_rejected',command_id=command['command_id'],error=str(exc)); continue
                if command['op'] not in ('request','request_plan','enqueue'): release(command['op'])
                if halted(): break
                task=state.apply(command,inventory_counts(obs))
                watchdog.reset()
                restored_terminal=False
                checkpoint()
                if command['op'] in ('request','modify','request_plan','modify_plan','resume','enqueue','reprioritize'): prepared=None
                emit('command_acknowledged',command_id=command['command_id'],op=command['op'],task=task,
                     observation_sequence=sequence,inventory=inventory_counts(obs))
            if halted(): break
            if scheduled:
                previous=state.current_id
                selected=state.select(inventory_counts(obs))
                if previous!=state.current_id:
                    if previous is not None: release('goal_switch')
                    prepared=None; watchdog.reset()
                    emit('goal_selected',selection=selected,priorities=state.ledger.snapshot())
                # The previous policy observation is already durable. Avoid a
                # second fsync for unchanged selection and clock-only updates.
                # Commands, switches, each new observation and shutdown still save.
                if previous!=state.current_id or state.task is None or state.task['status']!='running':
                    checkpoint()
                if state.all_terminal() and not restored_terminal:
                    report['status']='task_completed' if all(s.task['status']=='completed' for s in state.states.values()) else 'cancelled'
                    break
                if halted() or bus.pending(): continue
            task=state.snapshot()
            if restored_terminal:
                time.sleep(.01); continue
            if task and task['status']=='cancelled': report['status']='cancelled'; break
            if task is None or task['status']=='paused':
                time.sleep(.01); continue
            observe(); task=state.snapshot()
            if task['status']=='completed':
                if scheduled: continue
                report['status']='task_completed'; break
            if bus.pending(): continue
            if pause_if_stalled():
                prepared=None; continue
            key=(task['task_id'],task['revision'])
            if prepared!=key:
                emit('policy_reset_started',task=task)
                policy.reset(task['policy_goal']); prepared=key
                if policy_memory is not None:
                    report['policy_memory']=policy_memory.on_reset(task,world_id)
                    emit('policy_memory_resume',**report['policy_memory'])
                emit('policy_reset_finished',task=task,condition=getattr(policy,'condition_metadata',None),worker_pid=report['worker_pid'])
                if halted() or bus.pending(): continue
                if recovery is not None:
                    if not getattr(env,'last_info',{}).get('workshop'):release('recovery_observation_initialize')
                    if halted() or bus.pending():continue
                    recovery.bind(world_id,task,env.last_info['workshop'],time.monotonic(),emit)
            if recovery_blocked():break
            if recovery is not None and recovery.needs_switch(env.last_info['workshop'],time.monotonic()):
                release('neural_skill_switch')
                if halted() or bus.pending():continue
                observe();task=state.snapshot()
                if task['status']!='running' or key!=(task['task_id'],task['revision']):continue
                recovery.observe(task,env.last_info['workshop'],{},time.monotonic(),emit)
                recovery.switch(env.last_info['workshop'],time.monotonic(),emit)
            if recovery_blocked():break
            account()
            token=bus.token(); cycle_start=time.monotonic(); attempts+=1
            frame=frames/('%06d.png'%attempts); save_screen(obs['pov'],frame)
            synchronize()
            if halted() or bus.pending(): continue
            emit('inference_started',attempt=attempts,task=task,generation=token)
            before=time.monotonic(); action=policy.act(obs['pov']); synchronize(); after=time.monotonic()
            if policy_memory is not None:policy_memory.inferred()
            inference.append((after-before)*1000)
            emit('inference_finished',attempt=attempts,task=task,generation=token,
                 call_started_s=before,call_finished_s=after)
            if halted() or bus.pending() or token!=bus.token():
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='control_changed_or_stop')
                prepared=None; continue
            if recovery is not None and recovery.needs_switch(env.last_info['workshop'],time.monotonic()):
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='neural_skill_budget')
                continue
            if pause_if_stalled():
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='no_progress')
                prepared=None; continue
            def dispatch():
                if halted(): return None
                def step():
                    emit('policy_dispatch_started',attempt=attempts,task=task,generation=token,action=action,
                         neural_skill=recovery.mode if recovery else None)
                    begin=time.monotonic(); result=env.step(action); end=time.monotonic()
                    emit('policy_dispatch_finished',attempt=attempts,task=task,generation=token,
                         call_started_s=begin,call_finished_s=end)
                    return result
                return gate.dispatch(step)
            accepted,packed=bus.dispatch(token,dispatch)
            if not accepted:
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='control_changed_at_dispatch')
                prepared=None; continue
            if packed is None or not packed[0]: break
            account(service=True)
            obs,reward,done,info=packed[1]; sequence+=1; report['steps']+=1
            raw_done=bool(done);server_dead=False
            if recovery is not None:obs=server_inventory_observation(obs,info)
            if recovery is not None:server_dead=server_terminal(info,'policy_transition',raw_done)
            done=bool(raw_done or server_dead)
            inventory=inventory_counts(obs)
            emit('inventory_observed',task=task,observation_sequence=sequence,inventory=inventory,
                 life_stats=obs.get('life_stats'),location_stats=obs.get('location_stats'),
                 done=done,raw_done=raw_done,frame=str(frame.relative_to(output)),controller='policy',
                 workshop=info.get('workshop') if recovery else None,
                 client_inventory=obs.get('client_inventory'))
            # A queued modification takes precedence over old-task completion.
            if not bus.pending(): observe()
            if recovery is not None:
                if done:recovery.end('terminal',time.monotonic(),emit)
                else:recovery.observe(state.snapshot(),info['workshop'],jsonable(action),time.monotonic(),emit)
            cycles.append((time.monotonic()-cycle_start)*1000)
            emit('cycle_timing',attempt=attempts,cycle_ms=cycles[-1])
            if done:
                complete=(all(s.task['status']=='completed' for s in state.states.values()) if scheduled
                          else state.snapshot()['status']=='completed')
                report['status']='task_completed' if complete and not server_dead else 'environment_done'
                break
        if report['status']=='initializing':
            report['status']='time_limit' if gate.reason=='time_limit' else 'cancelled'
        report['execution_valid']=report['steps']>0 and report['status'] in ('task_completed','time_limit','environment_done','recovery_blocked')
    except BaseException as exc:
        report['status']='cancelled' if isinstance(exc,KeyboardInterrupt) else 'error'
        report['error']=repr(exc)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        control_ended=time.monotonic(); gate.stop(report['status'])
        try: account()
        except Exception as exc:
            report['schedule_accounting_error']=repr(exc); report['execution_valid']=False
        if inbox:
            try: inbox.close()
            except BaseException as exc: report['inbox_close_error']=repr(exc); report['execution_valid']=False
        emit('stop_acknowledged',reason=gate.reason,steps=report['steps'],task=state.snapshot())
        if recovery is not None:
            try:
                recovery.end('shutdown',time.monotonic(),emit)
                controller_state['neural_recovery']=recovery.state()
                report['neural_recovery']=dict(state=recovery.state(),source='local_rule_privileged_server',
                    threat_guard=recovery.threat_guard,
                    pursuit_guard=recovery.pursuit_guard,
                    skill_embedding_sha256=recovery.skill['sha256'],root_goal_preserved=True,direct_action_overrides=0,
                    llm_calls=0,max_ticks_per_skill=200,max_seconds_per_skill=30.)
                if recovery.threat_guard:
                    report['neural_recovery'].update(max_ticks_per_hazard_skill=400,max_seconds_per_hazard_skill=90.)
                if recovery.pursuit_guard:
                    report['neural_recovery'].update(max_ticks_per_hazard_skill=900,max_seconds_per_hazard_skill=180.)
                if recovery.hazard_skill is not recovery.skill:
                    report['neural_recovery']['experimental_hazard_condition']=recovery.hazard_skill['provenance']
            except Exception as exc:
                report['recovery_close_error']=repr(exc);report['execution_valid']=False;report['status']='error'
        try: checkpoint(force=True)
        except Exception as exc:
            report['checkpoint_error']=type(exc).__name__; report['execution_valid']=False
        try: report['release']=release('session_shutdown')
        except BaseException as exc: report['release']='failed: '+repr(exc); report['execution_valid']=False
        if recovery is not None:report['release_evidence']=getattr(env,'release_evidence',None)
        if world_store is not None and world_verified and report['status']!='error' and report['release']=='neutral_command_sent':
            try:
                observe(); checkpoint(force=True)
                saved_world=env.save_and_quit()
                memory_payload=None
                if policy_memory is not None:
                    try:
                        memory_payload=policy_memory.snapshot(output)
                        report['policy_memory']['saved']=memory_payload is not None
                    except Exception as exc:
                        report['policy_memory_error']=repr(exc);report['execution_valid']=False
                generation=world_store.commit(saved_world,state.checkpoint(world_id),obs,
                    **({'controller_state':controller_state} if controller_state else {}),
                    **({'policy_memory':memory_payload} if memory_payload else {}))
                report['world_save']={'status':'saved','generation':generation.name,'world_id':world_id}
                emit('world_saved',**report['world_save'])
            except Exception as exc:
                report['world_save']={'status':'failed','error':repr(exc)}; report['execution_valid']=False
                (output/'world-save-traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for label,resource in (('environment',env),('policy',policy)):
            try: resource.close()
            except BaseException as exc: report[label+'_close_error']=repr(exc); report['execution_valid']=False
        report.update(task=state.snapshot(),task_completed=bool(state.task and state.task['status']=='completed'),
                      elapsed_control_s=None if started is None else control_ended-started,
                      elapsed_cleanup_s=time.monotonic()-control_ended,
                      inference_ms=distribution(inference),control_cycle_ms=distribution(cycles),gpu=gpu_metrics())
        if scheduled:
            report.update(goals={k:s.snapshot() for k,s in state.states.items()},priorities=state.ledger.snapshot(),
                completed_goal_count=sum(s.task['status']=='completed' for s in state.states.values()),
                task_completed=bool(state.states) and all(s.task['status']=='completed' for s in state.states.values()))
        emit('episode_finished',status=report['status']); log.close()
        write_json(output/'summary.json',report)
    return report
