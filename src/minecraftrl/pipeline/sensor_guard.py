"""Pre-dispatch guards on the sensor path (sensor-tier design 2026-11-19 §2.5, user decision 11-19).

Every guard decides before the policy action reaches the game, from the `SensorFrame`, the seen map and the plan
(instruction) only. A guard only removes keys from the policy's action; it never makes an action of its own (AGENTS.md).

- Guard 1 (aim): the crosshair block, plus the block a camera turn in this action would aim at, cast on the seen map.
  An aim of kind 'other' or 'unknown' blocks the attack when the held tool is short for any guarded block (fail safe).
- Guard 2 `guard_drop_needed`: no drop while holding an item the plan needs, or together with a hotbar key.
- Guard 3 `guard_tool_wear`: hold before an attack when the held tool has fewer uses left than the plan still needs
  from it plus a margin; the correction is a spare-tool subgoal.
- Guard 4 `guard_hazard_move`: on the ground, no movement key toward a cell with lava or fire, an unknown floor, or a
  fall that leaves health at the low-health line or below; outside bridging and jumping, no damaging fall either.
"""
import math

import numpy as np

from .guard import VIOLATION_GAP_S, attacking
from .planner import block_family, harvest_verdict, tool_of

REACH = 4.5                         # survival block reach
TURN_DEG = 0.05                     # camera change below this keeps the crosshair as the only aim
LOW_HEALTH = 6.                     # same line as the low-health monitor
SAFE_FALL = 3                       # fall damage = blocks fallen - 3
WALK, SPRINT, HALF_WIDTH = 0.2158, 0.2806, 0.3    # blocks per tick; the feet box reaches 0.3 from the centre
PREDICT_TICKS = 2                   # design §2.5 guard 4 (my decision): the cells entered within two ticks
HAZARDS = ('lava', 'fire', 'soul_fire')
OPEN = ('air', 'water', 'unknown') + HAZARDS      # feet-level cells one can walk into (or does not know)
WEAR_MARGIN = 2                     # provisional; set from past records before S2 (design §2.5 guard 3)
MOVES = ('forward', 'back', 'left', 'right')


def pressed(action, key): return key in action and bool(np.asarray(action[key]).any())


def camera(action):
    if 'camera' not in action: return None
    value = np.asarray(action['camera'], dtype=float).reshape(-1)
    return (float(value[0]), float(value[1])) if value.size >= 2 else None


def look(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def move_direction(key, yaw):
    y = math.radians(yaw); fx, fz = -math.sin(y), math.cos(y)
    return {'forward': (fx, fz), 'back': (-fx, -fz), 'right': (-math.cos(y), -math.sin(y)), 'left': (math.cos(y), math.sin(y))}[key]


def planned_uses(steps, kb, context, tool):
    """How many targets the remaining plan steps still need this tool for (instruction: the plan and its knowledge)."""
    cls = tool_of(tool)[0]; total = 0
    for s in steps:
        if s.kind != 'policy' or not s.block: continue
        verdict, need = harvest_verdict(kb.query(block_family(s.block), 'harvest_requires', context=context), tool)
        if need is None or need['min_tier'] == 'hand' or need['tool_class'] is None: continue
        if verdict == 'ok' and cls == need['tool_class']: total += int(s.count)
    return total


def apply_masks(action, events):
    """(copy of the action with every masked key zeroed, masked keys in order)."""
    masked = []
    for e in events:
        for key in e.get('mask', ()):
            if key not in masked: masked.append(key)
    if not masked: return action, masked
    out = dict(action)
    for key in masked:
        if key in out: out[key] = np.zeros_like(np.asarray(out[key]))
    return out, masked


class SensorGuards:
    def __init__(self, guard, *, kb, context, needs, steps=(), skill_kind=None, wear_margin=WEAR_MARGIN, hold_s=10.):
        self.guard = guard; self.kb = kb; self.context = dict(context); self.needs = frozenset(needs)
        self.steps = list(steps); self.skill_kind = skill_kind; self.wear_margin = int(wear_margin); self.hold_s = hold_s
        self.since = {}

    def start_step(self, *, steps=None, skill_kind=None, needs=None):
        """A (re)started step: fresh persistence timers; the remaining plan steps, the step's skill kind, and the
        needed items of the current plan (a replan changes them)."""
        self.since.clear(); self.guard.reset()
        if steps is not None: self.steps = list(steps)
        if needs is not None: self.needs = frozenset(needs)
        self.skill_kind = skill_kind

    def _persist(self, kind, t):
        start, last = self.since.get(kind, (t, t))
        if t - last > VIOLATION_GAP_S: start = t
        self.since[kind] = (start, t)
        return t - start >= self.hold_s

    def _event(self, frame, kind, *, mask, hold=False, persistent=False, **fields):
        if persistent: kind += '_persistent'; hold = True
        return dict(kind=kind, hold=hold, mask=list(mask), phase='pre_dispatch', tick=frame.tick, **fields)

    def pre_dispatch(self, frame, seen, action, prev=None):
        if frame is None or frame.gui_open: return []
        return self.aim(frame, seen, action) + self.drop(frame, action) + self.wear(frame, action) + self.hazard(frame, seen, action, prev)

    # Guard 1 -------------------------------------------------------------------------------------------------------
    def short_for_any_guarded(self, held):
        for family in self.guard.guarded:
            verdict, _ = harvest_verdict(self.kb.query(family, 'harvest_requires', context=self.context), held)
            if verdict != 'ok': return True
        return False

    def aim(self, frame, seen, action):
        if not attacking(action): return []
        targets = []
        kind, cell = frame.crosshair
        if kind != 'air': targets.append(('crosshair', kind, cell))
        turn = camera(action)
        if turn is not None and max(abs(turn[0]), abs(turn[1])) > TURN_DEG:
            pitch = max(-90., min(90., frame.pitch + turn[0]))
            k, c = seen.cast(frame.eye, look(frame.yaw + turn[1], pitch), REACH)
            if k != 'air': targets.append(('ray_prediction', k, c))
        events = []
        for phase, kind, cell in targets:
            if kind in ('other', 'unknown'):
                if self.short_for_any_guarded(frame.selected_item):
                    # Counted apart: a block that needed no guarding may have been refused (design §2.5 fix 2).
                    events.append(dict(kind='guard_aim_uncertain', hold=False, mask=['attack'], phase=phase, block=kind,
                                       pos=cell, held=frame.selected_item, tick=frame.tick))
                continue
            event = self.guard.judge(frame, kind, cell, phase, 0)
            if event: events.append(dict(event, mask=['attack'] if event['mask_attack'] else []))
        return events

    # Guard 2 -------------------------------------------------------------------------------------------------------
    def drop(self, frame, action):
        if not pressed(action, 'drop'): return []
        hotbar = any(k.startswith('hotbar.') and pressed(action, k) for k in action)
        reason = 'hotbar_with_drop' if hotbar else 'needed_item' if frame.selected_item in self.needs else None
        if reason is None: return []
        return [self._event(frame, 'guard_drop_needed', mask=['drop'], reason=reason, held=frame.selected_item,
                            persistent=self._persist('guard_drop_needed', frame.t))]

    # Guard 3 -------------------------------------------------------------------------------------------------------
    def wear(self, frame, action):
        if not attacking(action) or frame.tool_max <= 0: return []
        planned = planned_uses(self.steps, self.kb, self.context, frame.tool_item)
        remaining = frame.tool_max - frame.tool_damage
        if planned <= 0 or remaining >= planned + self.wear_margin: return []
        return [self._event(frame, 'guard_tool_wear', mask=['attack'], hold=True, tool=frame.tool_item, remaining=remaining,
                            planned=planned, margin=self.wear_margin, correction='craft_spare_tool')]

    # Guard 4 -------------------------------------------------------------------------------------------------------
    def hazard_of(self, frame, seen, cell):
        x, y, z = cell
        at, below = seen.kind(cell), seen.kind((x, y - 1, z))
        if at in HAZARDS or below in HAZARDS: return 'lava_or_fire'
        if at not in OPEN: return None                    # a wall: walking into it is no fall
        fall = seen.depth_below(cell)
        if fall is None: return 'unknown_floor'
        depth, landing = fall
        if landing in HAZARDS: return 'lava_or_fire'
        damage = 0 if landing == 'water' else max(0, depth - SAFE_FALL)
        if frame.health - damage <= LOW_HEALTH: return 'lethal_fall'
        if damage > 0 and self.skill_kind not in ('bridge', 'jump'): return 'damaging_fall'
        return None

    def hazard(self, frame, seen, action, prev=None):
        if not frame.on_ground: return []                 # already falling or sliding: cannot be prevented here
        px, py, pz = frame.position
        here = (math.floor(px), math.floor(py + 1e-6), math.floor(pz))
        vx = vz = 0.
        if prev is not None and frame.tick > prev.tick:
            dt = frame.tick - prev.tick; vx = (px - prev.position[0]) / dt; vz = (pz - prev.position[2]) / dt
        events = []
        for key in MOVES:
            if not pressed(action, key): continue
            dx, dz = move_direction(key, frame.yaw)
            speed = SPRINT if key == 'forward' and pressed(action, 'sprint') else WALK
            reach = HALF_WIDTH + PREDICT_TICKS * (speed + max(0., vx * dx + vz * dz))
            cell = (math.floor(px + dx * reach), here[1], math.floor(pz + dz * reach))
            if cell == here: continue
            reason = self.hazard_of(frame, seen, cell)
            if reason is None: continue
            mask = [key] + [k for k in ('sprint', 'jump') if pressed(action, k)]
            events.append(self._event(frame, 'guard_hazard_move', mask=mask, reason=reason, key=key, cell=cell,
                                      health=frame.health, persistent=self._persist('guard_hazard_move', frame.t)))
        return events
