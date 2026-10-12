"""Explicit student-visited-state demonstrations, never a silent policy fallback."""
from copy import deepcopy
import math
import time

from .correction_teacher import CorrectionTeacher
from .rl_collect import RLEnvironment
from .rl_core import LOGS
from .runtime import jsonable, run_episode, write_json
from .stall_detection import StallDetector
from .stall_replay import wood_sample


def recovery_waypoints(world, gui):
    """Enter the known fixture corridor from the actual pose, without teleporting."""
    ox, oz = (v * 16 for v in world['chunk'])
    x, z = gui['server_x'] - ox, gui['server_z'] - oz
    if world['curriculum']['mirror_z']:
        z = 16 - z
    gap = world['curriculum']['wall_end'] + 1.5
    if not all(math.isfinite(v) for v in (x, z)) or not (1.29 <= x <= 14.71 and 1.29 <= z <= 14.71):
        raise ValueError('student outside supported recovery arena')
    if x <= 6.71:
        return [(5.5, z), (5.5, gap), (9.5, gap), (9.5, 3.5)]
    if x >= 8.29:
        return [(9.5, z), (9.5, 3.5)]
    if z >= gap - .2:
        return [(9.5, gap), (9.5, 3.5)]
    raise ValueError('student pose intersects the known barrier')


class StudentCorrection:
    """A single irreversible student -> neutral -> teacher handoff per episode.

    Only committed expert actions are BC labels. The full prefix and neutral
    transition remain mandatory recurrent context. STOP is owned by run_episode;
    this object never rearms it or dispatches actions on its own.
    """
    def __init__(self, student, raw, world, output, is_stopped=lambda: False):
        self.student, self.raw, self.world, self.output = student, raw, world, output
        self.teacher = CorrectionTeacher(raw, world)
        self.is_stopped = is_stopped
        self.rows, self.decisions, self.handoffs = [], [], []
        self.environment = None
        self.owner = 'student'
        self.pending = None
        self.trigger = None
        self.terminal = False
        self.student_success = False

    def set_reward_context(self, context):
        self.context = list(context)
        self.student.set_reward_context(context)
        self.teacher.set_reward_context(context)

    def reset(self, goal=None):
        self.student.reset(goal)
        self.teacher.reset(goal)
        self.teacher.environment = self.environment
        self.detector = StallDetector()
        self.detector.update(wood_sample(self.environment.gui, {}, str(self.output)))

    def act(self, screen):
        if self.is_stopped() or self.terminal:
            raise InterruptedError('STOP/terminal before correction inference')
        if self.pending is not None:
            raise RuntimeError('uncommitted correction action')
        before = deepcopy(self.environment.gui)
        owner = self.owner
        if owner == 'student':
            action = self.student.act(screen)
        elif owner == 'handoff_neutral':
            action = self.raw.action_space.noop()
        elif owner == 'teacher':
            action = self.teacher.act(screen)
        else:
            raise RuntimeError('invalid correction controller')
        self.controller_name = owner
        self.pending = dict(owner=owner, action=jsonable(action), server_before=jsonable(before),
                            reward_context=list(self.context), frame=f'frames/{len(self.rows):06d}.png',
                            loss_eligible=owner == 'teacher')
        return action

    def commit(self, action, reward, terminal):
        if self.pending is None or jsonable(action) != self.pending['action']:
            raise ValueError('correction action/dispatch mismatch')
        row = self.pending
        row.update(server_after=jsonable(self.environment.gui), reward=reward, terminal=terminal)
        self.rows.append(row)
        self.pending = None
        self.terminal = terminal
        if row['owner'] == 'student':
            self.student.commit(action, reward, terminal)
            self.student_success = bool(self.environment.reward.completed)
            decision = self.detector.update(wood_sample(self.environment.gui, jsonable(action), str(self.output),
                previous=row['server_before'], terminal=terminal))
            decision['step'] = len(self.rows)
            self.decisions.append(decision)
            if decision['event'] == 'stalled' and not terminal:
                self.trigger = deepcopy(decision)
                self.owner = 'handoff_neutral'
        elif row['owner'] == 'handoff_neutral':
            gui = self.environment.gui
            # Count this transition even if its proof fails; never start the teacher.
            verified = gui.get('server_available') == 1 and gui.get('error') == 0 and gui.get('held_key_count') == 0
            receipt = dict(step=len(self.rows), trigger=self.trigger, verified=verified,
                           held_key_count=gui.get('held_key_count'), server_tick=gui.get('server_tick'),
                           teacher_started=False)
            self.handoffs.append(receipt)
            if not verified:
                raise ValueError('handoff neutral keys not verified')
            if terminal or self.is_stopped():
                return
            if gui.get('screen') != 0:
                raise ValueError('handoff with an open GUI is unsupported')
            self.teacher.waypoints = recovery_waypoints(self.world, gui)
            receipt.update(teacher_started=True, waypoints=self.teacher.waypoints.copy())
            self.owner = 'teacher'
            self.detector.stop()
        else:
            self.teacher.commit(action, reward, terminal)
            row['phase'] = self.teacher.rows[-1]['phase']

    def close(self):
        pass  # Save source weights/committed prefix before disposing the model.


def run_student_correction(policy, env, output, *, world, split, gate, stop_file, **kwargs):
    stopped = lambda: gate.requested.is_set() or bool(stop_file and stop_file.exists())
    wrapper = RLEnvironment(env, policy, strict_ticks=True, reward_mode='wood-potential-v4', is_stopped=stopped)
    policy.environment = wrapper
    policy.is_stopped = stopped
    try:
        report = run_episode(policy, wrapper, output, goal='collect wood', target_items=LOGS,
            completion_check=lambda obs: wrapper.reward.completed, gate=gate, stop_file=stop_file, **kwargs)
        if report['execution_valid'] and report['release'] == 'neutral_command_sent':
            write_json(output/'rl-control-finished.json', dict(control_closed=True, release=report['release'], monotonic_s=time.monotonic()))
        release = getattr(env, 'release_evidence', {})
        expert_rows = sum(r['loss_eligible'] for r in policy.rows)
        completed = bool(wrapper.reward and wrapper.reward.completed)
        report.update(controller='explicit_student_to_privileged_teacher', learned_policy_loaded=True,
            rl_split=split, rl_training_eligible=False, rl_updates=0, initial_world=world,
            server_target_reached=completed, student_target_reached=policy.student_success,
            teacher_assisted_target_reached=completed and expert_rows > 0,
            demonstration_candidate=bool(report['execution_valid'] and completed and expert_rows > 0 and
                release.get('verified') and release.get('held_key_count') == 0),
            bc_training_ready=False, expert_labels=expert_rows, context_only_rows=len(policy.rows)-expert_rows,
            handoffs=policy.handoffs, server_tick_deltas=wrapper.ticks,
            teacher_observations='privileged server position, ray hit, known fixture geometry; not video-only',
            condition=getattr(policy.student, 'condition_metadata', {}))
        write_json(output/'correction-rows.json', policy.rows)
        write_json(output/'stall-decisions.json', policy.decisions)
        write_json(output/'initial-state.json', getattr(wrapper, 'initial', {}))
        write_json(output/'last-observed-state.json', dict(server=wrapper.gui))
        if hasattr(policy.student, 'initial_state'):
            from .rl_policy import cpu_tree
            policy.student.torch.save(dict(scope='correction_student_context_only', eligible=False,
                split=split, rows=policy.student.rows, initial_heads=policy.student.initial_heads,
                initial_recurrent_state=policy.student.initial_state,
                final_recurrent_state=cpu_tree(policy.student.agent.hidden_state),
                condition=policy.student.condition_metadata), output/'student-prefix.pt')
        write_json(output/'summary.json', report)
        return report
    finally:
        policy.student.dispose()
