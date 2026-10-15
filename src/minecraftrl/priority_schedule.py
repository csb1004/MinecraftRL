"""Deterministic, persistent priority aging. No rewards, clocks or game dispatch.

Callers supply active-session elapsed time and verified execution service separately.
Only the caller may assert completion, pause, blockage or a safe handoff boundary.
"""
from copy import deepcopy
import math


STATUSES = ('pending', 'blocked', 'paused', 'completed', 'cancelled')
DEFAULTS = dict(quantum_s=10., max_wait_s=60., age_step_s=30., age_gain=5.,
                age_cap=30., deadline_window_s=30., deadline_cap=30.)


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def priority_value(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError('priority must be an integer in [0,100]')
    return value


class PriorityLedger:
    def __init__(self, **settings):
        if set(settings) - set(DEFAULTS): raise ValueError('unknown scheduling setting')
        self.settings = dict(DEFAULTS, **settings)
        if any(not nonnegative(v) or v == 0 for v in self.settings.values()):
            raise ValueError('scheduling settings must be finite and positive')
        self.records = {}
        self.active = None
        self.slice_s = 0.

    def register(self, goal_id, *, priority=50, depends_on=(), deadline_s=None):
        priority_value(priority)
        if (not isinstance(goal_id, str) or not 1 <= len(goal_id) <= 256 or
                goal_id in self.records or len(self.records) >= 64):
            raise ValueError('invalid, duplicate or excessive goal identity')
        if not isinstance(depends_on, (list, tuple)) or any(
                not isinstance(d, str) or d not in self.records for d in depends_on):
            raise ValueError('dependencies must be existing goals')
        if len(depends_on) != len(set(depends_on)): raise ValueError('duplicate dependency')
        if deadline_s is not None and (not nonnegative(deadline_s) or deadline_s == 0):
            raise ValueError('deadline is a positive active-session lifetime')
        self.records[goal_id] = dict(priority=priority, depends_on=list(depends_on),
            deadline_s=deadline_s, status='pending', age_s=0., wait_s=0.,
            service_s=0., direct_service_s=0., service_credit_s=0.)

    def reprioritize(self, goal_id, priority):
        self.records[goal_id]['priority'] = priority_value(priority)

    def dependencies(self, graph):
        """Atomically rebind a validated plan without replacing timing identities."""
        trial = deepcopy(self.records)
        for key, deps in graph.items():
            if key not in trial or not isinstance(deps, list) or any(
                    not isinstance(d, str) or d not in trial for d in deps):
                raise ValueError('unknown dependency')
            if len(deps) != len(set(deps)): raise ValueError('duplicate dependency')
            trial[key]['depends_on'] = list(deps)
        self._validate_graph(trial)
        self.records = trial

    @staticmethod
    def _validate_graph(records):
        visiting, visited = set(), set()
        def visit(key):
            if key in visiting: raise ValueError('cyclic scheduling dependencies')
            if key in visited: return
            visiting.add(key)
            for dep in records[key]['depends_on']:
                if dep not in records: raise ValueError('missing dependency')
                visit(dep)
            visiting.remove(key); visited.add(key)
        for key in records: visit(key)

    def set_status(self, goal_id, status):
        if status not in STATUSES: raise ValueError('invalid scheduling status')
        record = self.records[goal_id]
        if record['status'] in ('completed', 'cancelled') and record['status'] != status:
            raise ValueError('terminal goal cannot be rearmed')
        record['status'] = status
        if self.active == goal_id and status != 'pending':
            self.active = None; self.slice_s = 0.

    def _frontiers(self, key):
        row = self.records[key]
        if row['status'] != 'pending': return set()
        remaining = [d for d in row['depends_on'] if self.records[d]['status'] != 'completed']
        if not remaining: return {key}
        return set().union(*(self._frontiers(d) for d in remaining))

    def advance(self, seconds, *, serving=None, service_s=0.):
        if (not nonnegative(seconds) or not nonnegative(service_s) or service_s > seconds or
                (serving is None and service_s) or (serving is not None and serving not in self.records)):
            raise ValueError('invalid active-session/service interval')
        frontiers = {key: self._frontiers(key) for key in self.records}
        if service_s and frontiers[serving] != {serving}:
            raise ValueError('service requires a dependency-ready goal')
        for key, row in self.records.items():
            if row['status'] not in ('pending', 'blocked'): continue
            row['age_s'] += seconds
            credited = service_s if serving in frontiers[key] else 0.
            if frontiers[key]: row['wait_s'] += seconds - credited
            row['service_s'] += credited
            if key == serving: row['direct_service_s'] += credited
            row['service_credit_s'] += credited
            if row['service_credit_s'] >= self.settings['quantum_s']:
                row['wait_s'] = 0.
                row['service_credit_s'] %= self.settings['quantum_s']
        if serving == self.active: self.slice_s += service_s

    def scores(self):
        result = {}
        for key, row in self.records.items():
            age = min(self.settings['age_cap'], row['age_s'] / self.settings['age_step_s'] * self.settings['age_gain'])
            deadline = 0.
            if row['deadline_s'] is not None:
                remaining = row['deadline_s'] - row['age_s']
                deadline = self.settings['deadline_cap'] * max(0., min(1.,
                    1. - remaining / self.settings['deadline_window_s']))
            result[key] = dict(base=row['priority'], age_boost=age, deadline_boost=deadline,
                               priority=row['priority'] + age + deadline)
        return result

    def select(self, *, safe_boundary=True):
        scores = self.scores()
        frontiers = {key: self._frontiers(key) for key in self.records}
        ready = [key for key, ends in frontiers.items() if ends == {key}]
        choices = {}
        for key in ready:
            parents = [p for p, ends in frontiers.items() if key in ends]
            source = max(parents, key=lambda p: (scores[p]['priority'], p))
            choices[key] = dict(goal_id=key, effective_priority=scores[source]['priority'],
                own_priority=scores[key]['priority'],
                inherited_from=source if source != key else None,
                wait_s=max(self.records[p]['wait_s'] for p in parents))
        if not choices:
            self.active = None; self.slice_s = 0.
            return dict(goal_id=None, reason='no_ready_goal')
        if self.active in choices and (not safe_boundary or self.slice_s < self.settings['quantum_s']):
            return dict(choices[self.active], reason='service_slice')
        if not safe_boundary: return dict(goal_id=None, reason='unsafe_handoff')
        overdue = [c for c in choices.values() if c['wait_s'] >= self.settings['max_wait_s']]
        if overdue:
            chosen = min(overdue, key=lambda c: (-c['wait_s'], -c['effective_priority'], -c['own_priority'], c['goal_id']))
            reason = 'maximum_wait'
        else:
            chosen = min(choices.values(), key=lambda c: (-c['effective_priority'], -c['wait_s'], -c['own_priority'], c['goal_id']))
            reason = 'priority'
        self.active = chosen['goal_id']; self.slice_s = 0.
        return dict(chosen, reason=reason)

    def snapshot(self):
        return dict(schema=1, settings=deepcopy(self.settings), records=deepcopy(self.records))

    @classmethod
    def restore(cls, value):
        if not isinstance(value, dict) or set(value) != {'schema', 'settings', 'records'} or value['schema'] != 1:
            raise ValueError('invalid priority checkpoint')
        result = cls(**value['settings'])
        records = value['records']
        if not isinstance(records, dict) or len(records) > 64: raise ValueError('invalid priority records')
        for key, row in records.items():
            if not isinstance(row, dict) or set(row) != {'priority', 'depends_on', 'deadline_s', 'status',
                    'age_s', 'wait_s', 'service_s', 'direct_service_s', 'service_credit_s'}:
                raise ValueError('invalid priority record')
            result.register(key, priority=row['priority'], deadline_s=row['deadline_s'])
            if row['status'] not in STATUSES or not isinstance(row['depends_on'], list) or any(
                    not isinstance(d, str) for d in row['depends_on']): raise ValueError('invalid goal state')
            if any(not nonnegative(row[k]) for k in ('age_s', 'wait_s', 'service_s', 'direct_service_s', 'service_credit_s')):
                raise ValueError('invalid goal timing')
            if (row['wait_s'] > row['age_s'] or row['service_s'] > row['age_s'] or
                    row['direct_service_s'] > row['service_s'] or
                    row['service_credit_s'] >= result.settings['quantum_s']):
                raise ValueError('inconsistent goal timing')
        result.dependencies({k: r['depends_on'] for k, r in records.items()})
        result.records = deepcopy(records)
        return result
