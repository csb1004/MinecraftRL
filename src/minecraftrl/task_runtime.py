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
                     synchronize=lambda:None, gpu_metrics=lambda:{},world_store=None,restore_bundle=None):
    if not 0<limit_s<=180: raise ValueError('limit_s must be in (0,180]')
    watchdog=ProgressWatchdog(no_progress_s)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    frames=output/'frames'; frames.mkdir(exist_ok=True)
    external_bus=bus is not None; bus=bus or CommandBus(); gate=gate or StopGate()
    state=TaskState(); inbox=None; obs=None; sequence=0; started=None
    world_id=uuid.uuid4().hex; saved_task=object()
    world_verified=False
    restored_terminal=False
    prepared=None; attempts=0; inference=[]; cycles=[]
    report={'status':'initializing','execution_valid':False,'steps':0,'environment_resets':0,
            'worker_pid':os.getpid(),
            'seed':seed,'controller':'policy','session_kind':'revisable_collection',
            'release':'not_attempted','no_progress_s':no_progress_s,'no_progress_events':0}
    log=(output/'events.jsonl').open('w',encoding='utf-8'); log_lock=threading.RLock()
    def emit(event, **fields):
        with log_lock:
            log.write(json.dumps(jsonable(dict(event=event,monotonic_s=time.monotonic(),**fields)),ensure_ascii=False,allow_nan=False)+'\n')
            log.flush()
    def halted():
        if stop_file and Path(stop_file).exists(): gate.stop('stop_file')
        if started is not None and time.monotonic()-started>=limit_s: gate.stop('time_limit')
        return gate.requested.is_set()
    def checkpoint(force=False):
        nonlocal saved_task
        task=state.snapshot()
        if force or task!=saved_task:
            atomic_json(output/'task-checkpoint.json',state.checkpoint(world_id))
            saved_task=task
    def release(reason):
        nonlocal obs,sequence
        outcome=env.release()
        # The neutral step advances MineRL: never infer from the pre-release frame.
        if getattr(env,'last_observation',None) is not None:
            obs=env.last_observation; sequence+=1
            inventory_counts(obs)
        emit('input_release',controller='cleanup',reason=reason,outcome=outcome,task=state.snapshot())
        return outcome
    def observe():
        if sequence<=state.last_observation: return
        event=state.observe(inventory_counts(obs),sequence)
        checkpoint()
        if event:
            name=event.pop('event'); emit(name,**event,inventory=inventory_counts(obs))
            if name=='plan_stage_completed': release('stage_transition')
    def pause_if_stalled():
        elapsed=watchdog.sample(state.snapshot(),time.monotonic())
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
            report['world_restore']=validate_restored_observation(restore_bundle,obs)
            world_id=restore_bundle['world_id']
            state=TaskState.restore(restore_bundle['task_checkpoint'],world_id)
            restored_terminal=bool(state.task and state.task['status'] in ('completed','cancelled'))
            sequence=state.last_observation+1
        world_verified=True
        baseline=inventory_counts(obs); started=time.monotonic()
        checkpoint(force=True)
        emit('episode_started',seed=seed,session_kind='revisable_collection',baseline_inventory=baseline,world_id=world_id,worker_pid=report['worker_pid'])
        if restore_bundle is not None:
            emit('world_restored',task=state.snapshot(),world_id=world_id,evidence=report['world_restore'])
        if not external_bus:
            inbox=DirectoryInbox(output/'commands',bus,emit); inbox.start()
        emit('session_ready',commands=str(output/'commands'))
        while not halted():
            if inbox and inbox.error: raise RuntimeError('command reader failed: '+inbox.error)
            for command in bus.drain():
                if halted(): break
                try: state.check(command)
                except ValueError as exc:
                    emit('command_rejected',command_id=command['command_id'],error=str(exc)); continue
                if command['op'] not in ('request','request_plan'): release(command['op'])
                if halted(): break
                task=state.apply(command,inventory_counts(obs))
                watchdog.reset()
                restored_terminal=False
                checkpoint()
                if command['op'] in ('request','modify','request_plan','modify_plan','resume'): prepared=None
                emit('command_acknowledged',command_id=command['command_id'],op=command['op'],task=task,
                     observation_sequence=sequence,inventory=inventory_counts(obs))
            if halted(): break
            task=state.snapshot()
            if restored_terminal:
                time.sleep(.01); continue
            if task and task['status']=='cancelled': report['status']='cancelled'; break
            if task is None or task['status']=='paused':
                time.sleep(.01); continue
            observe(); task=state.snapshot()
            if task['status']=='completed': report['status']='task_completed'; break
            if bus.pending(): continue
            if pause_if_stalled():
                prepared=None; continue
            key=(task['task_id'],task['revision'])
            if prepared!=key:
                emit('policy_reset_started',task=task)
                policy.reset(task['policy_goal']); prepared=key
                emit('policy_reset_finished',task=task,condition=getattr(policy,'condition_metadata',None),worker_pid=report['worker_pid'])
                if halted() or bus.pending(): continue
            token=bus.token(); cycle_start=time.monotonic(); attempts+=1
            frame=frames/('%06d.png'%attempts); save_screen(obs['pov'],frame)
            synchronize()
            if halted() or bus.pending(): continue
            emit('inference_started',attempt=attempts,task=task,generation=token)
            before=time.monotonic(); action=policy.act(obs['pov']); synchronize(); after=time.monotonic()
            inference.append((after-before)*1000)
            emit('inference_finished',attempt=attempts,task=task,generation=token,
                 call_started_s=before,call_finished_s=after)
            if halted() or bus.pending() or token!=bus.token():
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='control_changed_or_stop')
                prepared=None; continue
            if pause_if_stalled():
                emit('action_discarded',attempt=attempts,task=task,generation=token,reason='no_progress')
                prepared=None; continue
            def dispatch():
                if halted(): return None
                def step():
                    emit('policy_dispatch_started',attempt=attempts,task=task,generation=token,action=action)
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
            obs,reward,done,info=packed[1]; sequence+=1; report['steps']+=1
            inventory=inventory_counts(obs)
            emit('inventory_observed',task=task,observation_sequence=sequence,inventory=inventory,
                 life_stats=obs.get('life_stats'),location_stats=obs.get('location_stats'),
                 done=done,frame=str(frame.relative_to(output)),controller='policy')
            # A queued modification takes precedence over old-task completion.
            if not bus.pending(): observe()
            cycles.append((time.monotonic()-cycle_start)*1000)
            emit('cycle_timing',attempt=attempts,cycle_ms=cycles[-1])
            if done:
                report['status']='task_completed' if state.snapshot()['status']=='completed' else 'environment_done'
                break
        if report['status']=='initializing':
            report['status']='time_limit' if gate.reason=='time_limit' else 'cancelled'
        report['execution_valid']=report['steps']>0 and report['status'] in ('task_completed','time_limit','environment_done')
    except BaseException as exc:
        report['status']='cancelled' if isinstance(exc,KeyboardInterrupt) else 'error'
        report['error']=repr(exc)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        control_ended=time.monotonic(); gate.stop(report['status'])
        if inbox:
            try: inbox.close()
            except BaseException as exc: report['inbox_close_error']=repr(exc); report['execution_valid']=False
        emit('stop_acknowledged',reason=gate.reason,steps=report['steps'],task=state.snapshot())
        try: checkpoint(force=True)
        except Exception as exc:
            report['checkpoint_error']=type(exc).__name__; report['execution_valid']=False
        try: report['release']=release('session_shutdown')
        except BaseException as exc: report['release']='failed: '+repr(exc); report['execution_valid']=False
        if world_store is not None and world_verified and report['status']!='error' and report['release']=='neutral_command_sent':
            try:
                observe(); checkpoint(force=True)
                saved_world=env.save_and_quit()
                generation=world_store.commit(saved_world,state.checkpoint(world_id),obs)
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
        emit('episode_finished',status=report['status']); log.close()
        write_json(output/'summary.json',report)
    return report
