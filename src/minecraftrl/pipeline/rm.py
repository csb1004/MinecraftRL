"""Goal stage machine (reward-machine state) and the per-step observation sample the guard and monitors read."""
from dataclasses import dataclass

import numpy as np

from .steps import item_count


@dataclass(frozen=True)
class Sample:
    t: float
    tick: int
    valid: bool
    gui_open: bool
    position: tuple
    yaw: float
    pitch: float
    action_keys: tuple
    inventory: dict
    selected_item: str
    destroys: int
    last_destroyed: str
    last_destroyed_pos: tuple | None
    last_harvestable: bool
    mining_block: str
    mining_pos: tuple | None
    hit_block: str
    hit_pos: tuple | None
    health: float = 20.


def sample_from(info, action, t):
    w = info.get('workshop') or {}; d = info.get('demo') or {}
    valid = w.get('server_available') == 1 and not w.get('error') and d.get('demo_available') == 1
    position = tuple(float(w.get('server_' + a, 0.)) for a in 'xyz')
    mining = d.get('demo_mining_block', 'air'); hit = d.get('demo_hit_block', 'air')
    return Sample(t=t, tick=int(w.get('world_time', 0)), valid=bool(valid), gui_open=w.get('screen', 0) != 0, position=position,
                  yaw=float(w.get('yaw', 0.)), pitch=float(w.get('pitch', 0.)),
                  action_keys=tuple(sorted(k for k, v in action.items() if k != 'camera' and np.asarray(v).any())),
                  inventory={k: int(v) for k, v in (w.get('server_inventory') or {}).items() if v},
                  selected_item=w.get('server_selected_item') or 'air', destroys=int(d.get('demo_destroys', 0)),
                  last_destroyed=d.get('demo_last_destroyed_block', 'air'),
                  last_destroyed_pos=tuple(int(d['demo_last_destroy_' + a]) for a in 'xyz') if d.get('demo_destroys') else None,
                  last_harvestable=bool(d.get('demo_last_destroy_harvestable', 0)), mining_block=mining,
                  mining_pos=None if mining == 'air' else tuple(int(d['demo_mining_' + a]) for a in 'xyz'),
                  hit_block=hit, hit_pos=None if hit == 'air' else tuple(int(w.get('hit_' + a, 0)) for a in 'xyz'),
                  health=float(w.get('server_health', 20.)))


class StageMachine:
    """Policy steps complete on gain since activation (log = 6 kinds); rule steps on possession (+ equipped)."""

    def __init__(self, plan, inventory, *, goal_item='iron_ore', goal_count=1):
        self.goal_item = goal_item; self.goal_count = goal_count; self.history = []; self.inventory = dict(inventory)
        self.replace(plan, inventory)

    @property
    def current(self): return self.steps[self.index] if self.index < len(self.steps) else None

    @property
    def done(self):
        return self.index >= len(self.steps) or item_count(self.inventory, self.goal_item) >= self.goal_count

    def replace(self, plan, inventory):
        self.steps = list(plan.steps); self.index = 0; self.inventory = dict(inventory)
        self.base = item_count(inventory, self.current.target) if self.current else 0

    def observe(self, inventory, selected_item):
        self.inventory = dict(inventory); transitions = []
        while self.current:
            step = self.current
            if step.kind == 'rule':
                equip = step.params.get('equip')
                finished = inventory.get(step.target, 0) >= 1 and (equip is None or selected_item == equip)
                gained = {step.target: inventory.get(step.target, 0)}
            else:
                gain = item_count(inventory, step.target) - self.base
                finished = gain >= step.count; gained = {step.target: gain}
            if not finished: break
            transitions.append(dict(step_id=step.id, outcome='completed', gained=gained)); self.history.append(transitions[-1])
            self.index += 1
            if self.current: self.base = item_count(inventory, self.current.target)
        return transitions
