"""Task-independent result contract over existing, explicitly masked evidence."""
import math
from .runtime import jsonable
from .goal_spec import assert_goal

CONTEXT_FIELDS = (
    'locate', 'approach', 'interact', 'recover_result', 'inventory_result',
    'progress_fraction', 'remaining_fraction', 'quantity_log2', 'pending_result',
    'known', 'known_valid', 'reachable', 'reachable_valid',
    'no_progress_fraction', 'pending_fraction', 'last_attack', 'last_use',
    'last_move', 'last_camera', 'repeat_fraction', 'remaining_goals_fraction',
)
CONTEXT_DIM = len(CONTEXT_FIELDS)


def contract(queue):
    if queue.active is None:
        raise ValueError('semantic condition requires an executable active goal')
    goal = queue.goals[queue.active]
    record = queue.records[queue.active]
    unified = assert_goal(queue, queue.active)
    result = dict(schema='state_change_goal_v1', target=list(goal['items']),
                result='inventory_increase_from_activation', quantity=goal['count'],
                prerequisites=goal['requires'], objective=goal['objective'],
                baseline=record['baseline'], progress=record['progress'],
                completion_owner='caller', condition_source='fixed_observer',
                unsupported_facts='unknown')
    if unified is not None:
        result.update(goal_spec_sha256=queue.goal_spec['source_sha256'],
                      goal_contract_sha256=unified['sha256'], target_ref=unified['target_ref'])
    return result


def action_summary(action):
    if action is None:
        return [0.] * 4
    action = jsonable(action)
    def number(v):
        while isinstance(v, list): v = v[0]
        return float(v)
    camera = action.get('camera', [[0., 0.]])
    while len(camera) == 1 and isinstance(camera[0], list): camera = camera[0]
    return [float(number(action.get(k, 0)) > 0) for k in ('attack', 'use')] + [
        float(any(number(action.get(k, 0)) for k in ('forward', 'back', 'left', 'right', 'jump'))),
        min(1., sum(abs(float(v)) for v in camera) / 180.)]


def context(queue, phase, progress, evidence, last_action=None, repeat=0, last_progress_tick=None):
    description = contract(queue)
    goal = queue.goals[queue.active]
    tick = queue.last_tick
    snapshot = progress or {}
    pending = snapshot.get('pending_result_since')
    stage = {'search':'locate', 'approach':'approach', 'mine':'interact', 'collect':'recover_result'}.get(phase)
    if stage is None:
        raise ValueError('unsupported active stage')
    # A dirt macro labelled collect by S1 is not evidence of a dropped item.
    if goal['skill'] == 'dirt_fixed': stage = 'interact'
    fraction = min(1., description['progress'] / description['quantity'])
    values = [float(stage == k) for k in CONTEXT_FIELDS[:4]] + [1., fraction, 1.-fraction,
        math.log2(1.+description['quantity']) / math.log2(65.), float(pending is not None)]
    for name in ('target_known', 'target_reachable'):
        value = evidence.get(name)
        if value is not None and type(value) is not bool:
            raise ValueError('evidence must be bool or unknown')
        values += [float(value is True), float(value is not None)]
    last = snapshot.get('last_progress_tick', last_progress_tick)
    if last is None: last = queue.records[queue.active]['activated_tick']
    values += [min(1., max(0, tick-last) / 600.),
               0. if pending is None else min(1., max(0, tick-pending) / 300.)]
    values += action_summary(last_action)
    values += [min(1., repeat/120.), sum(r['status']!='completed' for r in queue.records.values())/len(queue.records)]
    if len(values) != CONTEXT_DIM or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError('invalid semantic context')
    return values, dict(description, stage=stage, evidence=evidence)


class ResultMonitor:
    """Failure evidence; repetition alone is deliberately insufficient to fail."""
    def __init__(self):
        self.goal = None
        self.last_action = None
        self.repeat = 0
        self.last_progress_tick = None
        self.peak = 0
        self.last_tick = None

    def observe(self, queue, action=None):
        if self.last_tick is not None and queue.last_tick <= self.last_tick:
            raise ValueError('monitor requires advancing authoritative ticks')
        self.last_tick = queue.last_tick
        # Completion/block clears queue.active before this final observation.
        # Keep the finished goal's evidence until another goal is selected.
        observed_goal = queue.active if queue.active is not None else self.goal
        if observed_goal != self.goal or self.last_progress_tick is None:
            self.goal = observed_goal
            self.peak = 0
            self.last_progress_tick = queue.last_tick
            self.repeat = 0
        normalized = None if action is None else jsonable(action)
        self.repeat = self.repeat+1 if normalized is not None and normalized==self.last_action else 0
        self.last_action = normalized
        if self.goal is not None:
            current = queue.records[self.goal]['progress']
            if current > self.peak:
                self.last_progress_tick = queue.last_tick
                self.peak = current
        return dict(no_inventory_progress_ticks=queue.last_tick-self.last_progress_tick,
                    exact_action_repeat=self.repeat, goal=self.goal,
                    repeat_is_failure=False)
