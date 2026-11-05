"""Rule monitors derived from the current goal (RM) and the active step (user rule, 2026-11-04).

Always anomalous: low health. Goal-derived: loss of a needed block (no drop) or needed item, and a blocked active
transition (e.g. no rise while climbing). Slowness is never an anomaly: stall reasons other than standing still
are recorded without hold.
"""
from collections import deque

from ..stall_detection import StallDetector, StallSample
from .planner import block_family, placeable, tool_of
from .steps import item_count

HOLD_KINDS = {'guard_unknown', 'guard_violation_persistent', 'destroy_not_harvestable', 'needed_tool_lost',
              'blocked_overhead', 'climb_without_block', 'stationary_repetition', 'low_health', 'rule_skill_failed'}
LOW_HEALTH = 6.
# Recorded escapes: open sky <= 7 jumps per 40 ticks without a block of height; ceiling 38-41 (detected in ~1-2 s).
BLOCKED_WINDOW_TICKS, BLOCKED_JUMPS = 40, 20


def inefficient(elapsed_s, baseline_s, factor=2.):
    return baseline_s is not None and elapsed_s > factor * baseline_s


class FastMonitor:
    def __init__(self, step, context_id, guarded, *, needs=frozenset(), drop_window_ticks=20):
        self.step = step; self.context_id = context_id; self.guarded = frozenset(guarded); self.needs = frozenset(needs)
        self.drop_window_ticks = drop_window_ticks; self.climb = step.params.get('expect') == 'rise' or step.params.get('until') == 'rise'
        self.prev = None; self.pending_drop = None; self.window = deque(); self.best_y = None; self.fired = set()
        self.start_y = None; self.closed_inventory = None  # last inventory seen with no GUI open
        self.stall = StallDetector()

    def event(self, kind, s, **evidence):
        return dict(kind=kind, hold=kind in HOLD_KINDS, tick=s.tick, step=self.step.id, evidence=evidence)

    def update(self, s):
        if not s.valid: return None
        prev, self.prev = self.prev, s
        events = []
        if s.health <= LOW_HEALTH and 'low_health' not in self.fired:
            self.fired.add('low_health'); events.append(self.event('low_health', s, health=s.health))
        removed = None
        if prev is not None and s.destroys > prev.destroys:
            removed = s.last_destroyed_pos
            if block_family(s.last_destroyed) in self.guarded:
                if not s.last_harvestable:
                    events.append(self.event('destroy_not_harvestable', s, block=s.last_destroyed, pos=s.last_destroyed_pos,
                                             held=prev.selected_item))
                else: self.pending_drop = (s.tick, item_count(s.inventory, self.step.target))
        if self.pending_drop:
            since, base = self.pending_drop
            if item_count(s.inventory, self.step.target) > base: self.pending_drop = None
            elif s.tick - since >= self.drop_window_ticks:
                self.pending_drop = None; events.append(self.event('harvest_without_drop', s, target=self.step.target))
        # Items in a crafting grid or on the cursor are not lost: compare GUI-closed inventories only (real check P3).
        base = self.closed_inventory
        if not s.gui_open: self.closed_inventory = s.inventory
        if base is not None and not s.gui_open and self.step.kind == 'policy':
            for item in sorted(self.needs):
                before, after = item_count(base, item), item_count(s.inventory, item)
                # A dropped or broken tool is lost; a placed block can be mined back (recorded only).
                kind = 'needed_tool_lost' if tool_of(item)[0] else 'needed_item_placed'
                if after < before: events.append(self.event(kind, s, item=item, before=before, after=after))
        if self.climb:
            if self.start_y is None: self.start_y = s.position[1]
            events.extend(self.blocked(s)); events.extend(self.handless(s))
        report = self.stall.update(StallSample(tick=s.tick, position=s.position, yaw=s.yaw, stage='search', action=s.action_keys,
                                               inventory_count=item_count(s.inventory, self.step.target), block_removed=removed,
                                               context_id=self.context_id))
        if report['event'] == 'stalled':
            events.append(self.event(report['reason'], s, no_progress_ticks=report.get('no_progress_ticks'),
                                     dominant_action=report.get('dominant_action')))
        return next((e for e in events if e['hold']), events[0] if events else None)

    def handless(self, s):
        """A pillar needs a block in hand: with more than one level to go an empty hand cannot rise (a last level can be
        stepped out of with a jump)."""
        remaining = self.step.count - (s.position[1] - self.start_y)
        if 'climb_without_block' in self.fired or placeable(s.selected_item) or remaining <= 1.: return []
        self.fired.add('climb_without_block')
        return [self.event('climb_without_block', s, held=s.selected_item, remaining=remaining)]

    def blocked(self, s):
        y = s.position[1]; best = y if self.best_y is None else self.best_y
        self.window.append((s.tick, y, 'jump' in s.action_keys))
        while self.window and s.tick - self.window[0][0] > BLOCKED_WINDOW_TICKS: self.window.popleft()
        self.best_y = max(best, y)
        ys = [w[1] for w in self.window]; jumps = sum(w[2] for w in self.window)
        if 'blocked_overhead' in self.fired or jumps < BLOCKED_JUMPS or max(ys) - min(ys) >= 1. or max(ys) >= best + 1.: return []
        self.fired.add('blocked_overhead')
        return [self.event('blocked_overhead', s, jumps=jumps, window_ticks=BLOCKED_WINDOW_TICKS, height_range=max(ys) - min(ys),
                           remaining=self.step.count - (y - self.start_y))]
