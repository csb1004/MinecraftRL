"""Typed execution feedback and stagnation (MineEvolve, arXiv 2603.13131, eqs. 1-3; design 2026-11-25-stagnation-remedy).

Progress over samples: p = (Var/(EPS_X + Var) + |dv|/(EPS_V + |dv|) + goal) / 3, with the horizontal position variance,
the server-inventory L1 change, and goal progress (the step's target count went up; for a climb, the height the player
stands at went up). A full window of WINDOW_TICKS with p < EPS_P is stagnation. The values come from the recorded
iron-flow runs (design §3): 2 of 25 completed attempts reach it, every time-limit failure does within 607-957 rows.
Inputs are server inventory and pose (game state for monitoring), never policy inputs.
"""
from collections import deque

from .steps import item_count

WINDOW_TICKS = 600
EPS_P = 0.1
EPS_X = 1.      # blocks^2
EPS_V = 1.      # items
FLOOR_TICKS = 40  # the lowest height over this long is where the player stands (a jump in place is no rise)
STRIDE_TICKS = 10  # how often a full window is scored
STALL_KINDS = ('stagnation', 'stationary_repetition')  # holds that mean the attempt stalled


def climbing(step): return step.params.get('until') == 'rise' or step.params.get('expect') == 'rise'


def variance(points):
    """Horizontal position variance (x and z); height is the climb's own goal term."""
    n = len(points)
    if n < 2: return 0.
    total = 0.
    for j in (0, 2):
        mean = sum(p[j] for p in points) / n; total += sum((p[j] - mean) ** 2 for p in points) / n
    return total


def l1(before, after): return sum(abs(int(after.get(k, 0)) - int(before.get(k, 0))) for k in set(before) | set(after))


def standing_rise(heights):
    """[(tick, y)] -> how much higher the player stands at the end than at the start."""
    first, last = heights[0][0], heights[-1][0]
    return min(y for t, y in heights if last - t < FLOOR_TICKS) - min(y for t, y in heights if t - first < FLOOR_TICKS)


def progress(samples, step):
    """[(tick, position, inventory)] -> dict(p, move, inventory, goal, variance, inventory_l1)."""
    var = variance([s[1] for s in samples]); dv = l1(samples[0][2], samples[-1][2])
    if climbing(step): goal = standing_rise([(s[0], s[1][1]) for s in samples]) >= .9
    else: goal = item_count(samples[-1][2], step.target) > item_count(samples[0][2], step.target)
    move, inventory = var / (EPS_X + var), dv / (EPS_V + dv)
    return dict(p=(move + inventory + float(goal)) / 3., move=move, inventory=inventory, goal=float(goal), variance=var,
                inventory_l1=dv)


class ProgressWindow:
    """Live stagnation for one policy-step attempt (MineEvolve eq. 3); fires at most once."""

    def __init__(self, step, window_ticks=WINDOW_TICKS, eps=EPS_P):
        self.step = step; self.window_ticks = window_ticks; self.eps = eps
        self.samples = deque(); self.first = None; self.scored = None; self.fired = False

    def update(self, tick, position, inventory):
        if self.fired: return None
        if self.first is None: self.first = tick
        self.samples.append((tick, tuple(position), dict(inventory)))
        while tick - self.samples[0][0] > self.window_ticks: self.samples.popleft()
        if tick - self.first < self.window_ticks or (self.scored is not None and tick - self.scored < STRIDE_TICKS): return None
        self.scored = tick
        score = progress(list(self.samples), self.step)
        if score['p'] >= self.eps: return None
        self.fired = True
        return dict(score, window_ticks=self.window_ticks, eps=self.eps)


class Attempt:
    """One execution of a policy step -> typed feedback e = (z, y, ds, dv, f, p, l) (MineEvolve eq. 1)."""

    def __init__(self, step, tick, position, inventory):
        self.step = step; self.start = (tick, tuple(position), dict(inventory)); self.last = self.start
        self.n = 1; self.sums = [position[0], position[2], position[0] ** 2, position[2] ** 2]
        self.low = [(tick, position[1])]   # recent heights, for the standing height at the end

    def add(self, tick, position, inventory):
        self.last = (tick, tuple(position), dict(inventory)); self.n += 1
        for i, v in enumerate((position[0], position[2], position[0] ** 2, position[2] ** 2)): self.sums[i] += v
        self.low.append((tick, position[1]))
        while tick - self.low[0][0] >= FLOOR_TICKS: self.low.pop(0)

    def finish(self, status, failure_type):
        (t0, p0, inv0), (t1, p1, inv1) = self.start, self.last
        n = self.n; var = sum(max(0., self.sums[2 + j] / n - (self.sums[j] / n) ** 2) for j in (0, 1))
        dv = l1(inv0, inv1); success = status == 'completed'
        if climbing(self.step): goal = success or min(y for _, y in self.low) - p0[1] >= .9
        else: goal = success or item_count(inv1, self.step.target) > item_count(inv0, self.step.target)
        p = (var / (EPS_X + var) + dv / (EPS_V + dv) + float(goal)) / 3.
        change = {k: int(inv1.get(k, 0)) - int(inv0.get(k, 0)) for k in set(inv0) | set(inv1)
                  if int(inv1.get(k, 0)) != int(inv0.get(k, 0))}
        return dict(step=self.step.id, skill=self.step.skill, target=self.step.target, success=success,
                    state_change=dict(start=list(p0), end=list(p1), height=p1[1] - p0[1]), inventory_change=change,
                    failure_type=None if success else failure_type, progress=p,
                    # Held for a stalled window, or no progress over the whole attempt (an attempt can move first, then stall).
                    stagnant=bool(not success and (p < EPS_P or failure_type in STALL_KINDS)), ticks=t1 - t0)
