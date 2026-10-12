"""Observation-only stagnation evidence; no reward, RM or action mutation.

Coordinates and inventory supplied by the replay adapter are privileged server
observations. A cycle is evidence of repetition, never proof of infeasibility.
"""
from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class StallSample:
    tick: int
    position: Tuple[float, float, float]
    yaw: float
    stage: str
    action: Tuple[str, ...]
    target: Optional[Tuple[int, int, int]] = None
    distance: Optional[float] = None
    inventory_count: int = 0
    mining_progress: Optional[float] = None
    block_removed: Optional[Tuple[int, int, int]] = None
    context_id: str = 'episode'
    valid: bool = True
    terminal: bool = False


def observed_cycles(nodes):
    """Extract actually traversed simple cycles via chronological loop erasure.

Rotations of a directed cycle share a key. Reversal does not. This does not
enumerate all cycles in a static graph, which would lose traversal order.
"""
    stack, positions, counts, witnesses = [], {}, Counter(), {}
    for node in nodes:
        if node in positions:
            start = positions[node]
            cycle = tuple(stack[start:])
            # Each simple cycle has unique nodes, so rotate its smallest node first.
            offset = min(range(len(cycle)), key=lambda i: repr(cycle[i]))
            key = cycle[offset:] + cycle[:offset]
            counts[key] += 1
            witnesses[key] = list(key) + [key[0]]
            for old in stack[start + 1:]:
                del positions[old]
            del stack[start + 1:]
        else:
            positions[node] = len(stack)
            stack.append(node)
    return [dict(nodes=witnesses[k], count=n) for k, n in counts.most_common()]


class StallDetector:
    """One instance per world/task revision; bounded to 20,000 observations.

Thresholds are initial development hypotheses in server ticks. Emitted events
are replan candidates only. Execution, capability checks and STOP belong to the
caller, not this detector.
"""
    windows = dict(search=400, approach=200, mine=400, collect=200, complete=200)
    goal_budget_ticks = 800
    history_ticks = 800
    max_samples = 20000

    def __init__(self):
        self.history = deque()
        self.visited = set()
        self.views = set()
        self.targets = set()
        self.removed = set()
        self.distances = {}
        self.damage = {}
        self.inventory_best = None
        self.last_tick = None
        self.last_progress = None
        self.last_goal_progress = None
        self.last_progress_reasons = []
        self.active = False
        self.ended = False
        self.context_id = None
        self.window = 0
        self.samples = 0

    def stop(self):
        self.ended = True
        self.active = False

    @staticmethod
    def _finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def _validate(self, s):
        if not self._finite(s.tick) or s.tick < 0 or int(s.tick) != s.tick:
            raise ValueError('invalid server tick')
        if not isinstance(s.context_id, str) or not s.context_id:
            raise ValueError('missing world/task revision context')
        if self.context_id is not None and s.context_id != self.context_id:
            raise ValueError('new world/task revision requires a new detector')
        if not s.valid:
            return
        if s.stage not in self.windows:
            raise ValueError('unsupported progress stage')
        if len(s.position) != 3 or not all(self._finite(x) for x in s.position) or not self._finite(s.yaw):
            raise ValueError('invalid position/yaw')
        for target in (s.target, s.block_removed):
            if target is not None and (not isinstance(target, tuple) or len(target) != 3 or
                                       not all(self._finite(v) and int(v) == v for v in target)):
                raise ValueError('invalid block identity')
        if not isinstance(s.action, tuple) or not all(isinstance(x, str) for x in s.action):
            raise ValueError('action signature must be a tuple of strings')
        if not self._finite(s.inventory_count) or s.inventory_count < 0 or int(s.inventory_count) != s.inventory_count:
            raise ValueError('invalid objective inventory count')
        if s.distance is not None and (s.target is None or not self._finite(s.distance) or s.distance < 0):
            raise ValueError('invalid target distance')
        if s.mining_progress is not None and (s.target is None or not self._finite(s.mining_progress) or
                                              not 0 <= s.mining_progress <= 1):
            raise ValueError('invalid observed mining progress')

    def _progress(self, s):
        reasons, objective = [], []
        cell = tuple(math.floor(x / 2.) for x in s.position)
        heading = math.floor(((s.yaw + 22.5) % 360) / 45.)
        if s.stage in ('search', 'approach'):
            if cell not in self.visited:
                reasons.append('new_position_cell')
            if s.stage == 'search' and (cell, heading) not in self.views:
                reasons.append('new_search_view')
        self.visited.add(cell)
        self.views.add((cell, heading))
        if s.target is not None:
            if s.target not in self.targets:
                objective.append('new_target')
                self.targets.add(s.target)
            if s.distance is not None:
                best = self.distances.setdefault(s.target, s.distance)
                if s.distance <= best - .5:
                    self.distances[s.target] = s.distance
                    objective.append('best_distance_improved')
            if s.mining_progress is not None:
                best = self.damage.get(s.target, 0.)
                if s.mining_progress >= best + .02:
                    self.damage[s.target] = s.mining_progress
                    objective.append('observed_mining_progress')
        if s.block_removed is not None and s.block_removed not in self.removed:
            self.removed.add(s.block_removed)
            objective.append('observed_block_disappearance')
        if self.inventory_best is None:
            self.inventory_best = s.inventory_count
        elif s.inventory_count > self.inventory_best:
            self.inventory_best = s.inventory_count
            objective.append('objective_inventory_increased')
        return reasons + objective, objective

    def update(self, s):
        if self.ended:
            raise RuntimeError('observation after STOP/terminal')
        self._validate(s)
        if self.samples >= self.max_samples:
            raise RuntimeError('detector observation budget exhausted')
        self.samples += 1
        self.context_id = s.context_id
        result = dict(tick=s.tick, event=None, stalled=False, reason=None,
                      progress=[], observation_valid=bool(s.valid))
        if s.terminal:
            self.stop()
            return dict(result, event='terminal')
        if not s.valid or (self.last_tick is not None and not 0 < s.tick - self.last_tick <= 4):
            self.history.clear()
            self.last_tick = s.tick
            self.last_progress = self.last_goal_progress = s.tick
            self.window = 0
            self.active = False
            self.last_progress_reasons = ['observation_gap']
            return dict(result, event='observation_gap', observation_valid=False)

        initial = self.last_tick is None
        self.last_tick = s.tick
        progress, objective = self._progress(s)
        if initial or progress:
            self.last_progress = s.tick
            self.window = 0
            self.last_progress_reasons = ['initial'] if initial else progress
        if initial or objective:
            self.last_goal_progress = s.tick
        self.window = max(self.window, self.windows[s.stage])
        node = (tuple(math.floor(x) for x in s.position),
                math.floor(((s.yaw + 22.5) % 360) / 45.), s.stage, s.target)
        self.history.append((s, node))
        while self.history and self.history[0][0].tick < s.tick - self.history_ticks:
            self.history.popleft()
        stalled = s.tick - self.last_progress >= self.window
        goal_expired = s.tick - self.last_goal_progress >= self.goal_budget_ticks
        result.update(progress=progress, no_progress_ticks=s.tick-self.last_progress,
                      no_goal_progress_ticks=s.tick-self.last_goal_progress,
                      window_ticks=self.window, last_progress_reasons=self.last_progress_reasons,
                      stalled=stalled or goal_expired)
        if result['stalled'] and not self.active:
            # Only use evidence from the current observation window.
            recent = [(a,n) for a,n in self.history if a.tick >= s.tick-self.window]
            nodes = [n for _,n in recent]
            # Remove dwell self-loops when measuring multi-node traversals.
            path = [n for i,n in enumerate(nodes) if i == 0 or n != nodes[i-1]]
            cycles = [c for c in observed_cycles(path) if c['count'] >= 2]
            action, count = Counter(a.action for a,_ in recent).most_common(1)[0]
            spans = [max(a.position[j] for a,_ in recent)-min(a.position[j] for a,_ in recent)
                     for j in range(3)]
            reason = 'no_goal_progress' if goal_expired and not stalled else 'no_progress'
            if stalled and max(spans) <= .5 and count / len(recent) >= .8:
                reason = 'stationary_repetition'
            elif stalled and cycles:
                reason = 'repeated_cycle'
            result.update(event='stalled', reason=reason,
                          cycle=cycles[0] if cycles else None,
                          graph_nodes=len(set(nodes)), graph_edges=len(set(zip(nodes,nodes[1:]))),
                          dominant_action=action, action_fraction=count/len(recent),
                          position_span=spans, evidence_start_tick=recent[0][0].tick)
        elif not result['stalled'] and self.active:
            result['event'] = 'progress_resumed'
        self.active = result['stalled']
        return result
