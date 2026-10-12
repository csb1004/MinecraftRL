"""Bounded, explicit goal-conditioning diagnostic; every action remains neural."""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np

from .rl_policy import state_fingerprint
from .runtime import jsonable, run_episode, write_json
from .stall_detection import StallDetector
from .stall_replay import wood_sample
from .world_store import digest


class GoalRecoveryController:
    max_intervention_ticks = 200

    def __init__(self, student, embedding_path, output, is_stopped=lambda: False):
        self.student, self.output = student, Path(output)
        self.embedding_path = Path(embedding_path)
        self.intervention_embed = np.load(self.embedding_path, allow_pickle=False)
        if (self.intervention_embed.shape != (1,512) or self.intervention_embed.dtype.kind != 'f'
                or not np.isfinite(self.intervention_embed).all()):
            raise ValueError('finite 1x512 goal embedding required')
        self.embedding_file_sha256 = digest(self.embedding_path)
        self.is_stopped = is_stopped
        self.environment = None
        self.rows, self.decisions, self.switches = [], [], []
        self.pending = self.scheduled = self.trigger = None
        self.ended = False
        self.mode = 'wood'

    def set_reward_context(self, context):
        self.context = list(context)
        self.student.set_reward_context(context)

    def reset(self, goal=None):
        if self.rows or self.switches:
            raise RuntimeError('goal recovery instance is single-episode')
        self.student.reset(goal)
        self.wood_embed = self.student.goal_embed.copy()
        if self.wood_embed.shape != self.intervention_embed.shape:
            raise ValueError('wood/intervention embedding shape mismatch')
        self.embed_hashes = {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in
                            (('wood',self.wood_embed), ('intervention',self.intervention_embed))}
        self.detector = StallDetector()
        self.detector.update(wood_sample(self.environment.gui, {}, str(self.output)))
        self.condition_metadata = dict(self.student.condition_metadata,
            intervention=dict(mode='one_stall_conditioning_probe', eligible_for_training=False,
                              embedding_path=str(self.embedding_path), embedding_file_sha256=self.embedding_file_sha256,
                              embedding_sha256=self.embed_hashes, max_ticks=self.max_intervention_ticks,
                              max_interventions=1, root_goal_preserved=True, memory_reset=False,
                              detector='privileged_server_graph_stall', direct_action_override=False))

    def act(self, screen):
        if self.ended or self.is_stopped():
            raise InterruptedError('STOP/terminal before goal recovery inference')
        if self.pending is not None:
            raise RuntimeError('uncommitted goal recovery action')
        before = deepcopy(self.environment.gui)
        if self.scheduled is not None:
            if before.get('screen') != 0:
                raise ValueError('goal switch with an open GUI is unsupported')
            mode, reason = self.scheduled
            memory_hash = state_fingerprint(self.student.agent.hidden_state)
            self.student.goal_embed = (self.intervention_embed if mode == 'intervention' else self.wood_embed).copy()
            assert state_fingerprint(self.student.agent.hidden_state) == memory_hash
            self.mode = mode
            if mode == 'intervention':
                self.intervention_start_tick = before['server_tick']
            self.switches.append(dict(after_step=len(self.rows), before_action=len(self.rows)+1,
                mode=mode, reason=reason, server_before=jsonable(before), memory_sha256=memory_hash,
                input_image_sha256=hashlib.sha256(np.asarray(screen).tobytes()).hexdigest(),
                input_image_shape=list(np.asarray(screen).shape), frame=f'frames/{len(self.rows):06d}.png',
                embedding_sha256=self.embed_hashes[mode], committed=False))
            self.scheduled = None
        action = self.student.act(screen)
        self.controller_name = 'steve1_'+self.mode
        self.pending = dict(mode=self.mode, owner=self.controller_name, action=jsonable(action),
                            server_before=jsonable(before), reward_context=list(self.context),
                            frame=f'frames/{len(self.rows):06d}.png',
                            embedding_sha256=self.embed_hashes[self.mode], loss_eligible=False)
        return action

    def commit(self, action, reward, terminal):
        if self.pending is None or jsonable(action) != self.pending['action']:
            raise ValueError('goal recovery action/dispatch mismatch')
        self.student.commit(action, reward, terminal)
        row = self.pending
        row.update(server_after=jsonable(self.environment.gui), reward=reward, terminal=terminal)
        self.rows.append(row)
        self.pending = None
        self.student.rows[-1]['goal_conditioning'] = dict(mode=row['mode'], embedding_sha256=row['embedding_sha256'])
        if self.switches and self.switches[-1]['before_action'] == len(self.rows):
            self.switches[-1]['committed'] = True
        if self.is_stopped():
            self.ended = True
            self.scheduled = None
            self.detector.stop()
            self.decisions.append(dict(step=len(self.rows), event='stop', tick=self.environment.gui['server_tick']))
            return
        sample = wood_sample(self.environment.gui, jsonable(action), str(self.output),
                             previous=row['server_before'], terminal=terminal)
        decision = self.detector.update(sample)
        decision['step'] = len(self.rows)
        self.decisions.append(decision)
        if terminal:
            self.ended = True
            self.scheduled = None
            return
        if self.mode == 'intervention':
            if sample.stage in ('mine','collect'):
                self.scheduled = ('wood','wood_interaction_ready')
            elif sample.tick-self.intervention_start_tick >= self.max_intervention_ticks:
                self.scheduled = ('wood','tick_budget')
        elif self.trigger is None and decision['event'] == 'stalled':
            self.trigger = deepcopy(decision)
            self.scheduled = ('intervention','graph_stalled')

    def close(self):
        self.ended = True
        if hasattr(self, 'detector'):
            self.detector.stop()


def run_goal_recovery(policy, env, output, *, world, split, gate, stop_file, **kwargs):
    """Diagnostic-only collector with the existing RM and STOP/process boundary."""
    from .rl_collect import RLEnvironment
    from .rl_core import LOGS
    from .rl_policy import cpu_tree
    if split != 'eval':
        raise ValueError('goal conditioning diagnostic is eval-only')
    output = Path(output)
    stopped = lambda: gate.requested.is_set() or bool(stop_file and stop_file.exists())
    wrapper = RLEnvironment(env, policy, strict_ticks=True, reward_mode='wood-potential-v4', is_stopped=stopped)
    policy.environment, policy.is_stopped = wrapper, stopped
    # Bind the actual loaded tensors to the preserved checkpoint without writing
    # another large copy of identical weights into every diagnostic trajectory.
    fingerprints = {name:state_fingerprint(value) for name,value in policy.student.initial_heads.items()}
    write_json(output/'initial-weights.json', dict(fingerprints=fingerprints,
               source_checkpoint=getattr(policy.student, 'checkpoint', None)))
    np.save(output/'intervention-embed.npy', policy.intervention_embed)
    def control_closed(report):
        # RLEnvironment.release has verified server key release and policy.close
        # has ended dispatch. Only cleanup/serialization may remain after this.
        write_json(output/'rl-control-finished.json',dict(control_closed=True,release=report['release'],
            cancelled_archive=report['status']=='cancelled',monotonic_s=time.monotonic(),
            scope='input release and policy closure verified; no further game actions'))
    try:
        report = run_episode(policy,wrapper,output,goal='collect wood',target_items=LOGS,
            completion_check=lambda obs: wrapper.reward.completed,gate=gate,stop_file=stop_file,
            on_control_closed=control_closed,**kwargs)
        condition = getattr(policy,'condition_metadata',{})
        report.update(controller='steve1_graph_goal_conditioning',learned_policy_loaded=True,
            rl_split='eval',rl_training_eligible=False,rl_updates=0,initial_world=world,
            server_target_reached=bool(wrapper.reward and wrapper.reward.completed),
            condition=condition,trigger=policy.trigger,switches=policy.switches,
            intervention_committed=any(r['mode']=='intervention' for r in policy.rows),
            mode_steps={mode:sum(r['mode']==mode for r in policy.rows) for mode in ('wood','intervention')},
            expert_labels=0,direct_action_overrides=0,server_tick_deltas=wrapper.ticks,
            simulation_ticks=sum(wrapper.ticks),neutral_startup_steps=60,
            nominal_simulation_seconds=sum(wrapper.ticks)/20.,initial_head_fingerprints=fingerprints,
            diagnostic_scope='Privileged stall trigger, neural actions, preserved wood RM; no training eligibility')
        if wrapper.reward is not None:
            report['reward_config'] = dict(wrapper.reward.metadata)
            report['reward_component_sums'] = {key:sum(r.get('reward_components',{}).get(key,0) for r in policy.rows)
                                              for key in ('task','shaping','time_cost','failure')}
        write_json(output/'goal-recovery-rows.json',policy.rows)
        write_json(output/'stall-decisions.json',policy.decisions)
        write_json(output/'goal-switches.json',policy.switches)
        write_json(output/'goal-conditioning-config.json',condition)
        write_json(output/'initial-state.json',dict(getattr(wrapper,'initial',{}),initial_world=world,seed=kwargs['seed']))
        write_json(output/'last-observed-state.json',dict(server=wrapper.gui))
        planning = hasattr(policy,'planner_events')
        if planning:
            report.update(controller='steve1_planned_goal_conditioning',actual_llm_calls=policy.provider.actual_llm_calls,
                planner_provider=policy.provider.kind,planner_calls=policy.provider.calls,
                planner_live=policy.provider.live,
                neutral_control_transitions=sum(r['owner']=='planner_neutral' for r in policy.rows),
                planner_phase=policy.phase,
                recovery_attempts=policy.attempts,
                diagnostic_scope='Bounded planning handoff; neural motor actions, no training or final-evaluation claim',
                llm_call_accounting='Windows broker api-receipt.json' if policy.provider.live else 'local fixture: no API calls')
            write_json(output/'planner-events.json',policy.planner_events)
            write_json(output/'resumed-stall-decisions.json',policy.resumed_decisions)
        if hasattr(policy.student,'initial_state'):
            assert len(policy.student.rows) == len(policy.rows)
            for saved,raw in zip(policy.student.rows,policy.rows):
                saved['reward_components'] = raw.get('reward_components')
                saved['frame'] = raw['frame']
            policy.student.torch.save(dict(schema=1,scope='planning_goal_conditioning_diagnostic' if planning else 'goal_conditioning_diagnostic',eligible=False,split='eval',
                rows=policy.student.rows,initial_head_fingerprints=fingerprints,
                source_checkpoint=getattr(policy.student,'checkpoint',None),
                initial_recurrent_state=policy.student.initial_state,
                final_recurrent_state=cpu_tree(policy.student.agent.hidden_state),
                condition=condition,terminal=wrapper.terminal,status=report['status'],
                reward_config=report.get('reward_config'),seed=kwargs['seed'],initial_world=world),output/'trajectory.pt')
        write_json(output/'summary.json',report)
        return report
    finally:
        policy.student.dispose()
