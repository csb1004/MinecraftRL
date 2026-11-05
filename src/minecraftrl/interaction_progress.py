"""Task-independent interaction/result memory over a fixed observation contract.

No motor actions, rewards, target coordinates or new sensors are supplied here.
Confirmed events outlive ephemeral target hints; completion remains caller-owned.
"""
import math


def count(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or value != int(value)):
        raise ValueError('invalid '+name)
    return int(value)


class InteractionProgress:
    def __init__(self, *, tick, interactions, inventory_count, stall_ticks=600, recovery_ticks=300,
                 known=False, reachable=False):
        self.last_tick = count(tick, 'tick')
        self.interactions = count(interactions, 'interactions')
        self.inventory_count = count(inventory_count, 'inventory_count')
        self.inventory_peak = self.inventory_count
        self.stall_ticks = count(stall_ticks, 'stall_ticks')
        self.recovery_ticks = count(recovery_ticks, 'recovery_ticks')
        if not self.stall_ticks or not self.recovery_ticks:
            raise ValueError('positive progress budgets required')
        self.last_progress_tick = self.last_tick
        self.pending_since = None
        if type(known) is not bool or type(reachable) is not bool or (reachable and not known):
            raise ValueError('invalid initial target evidence')
        self.stage = 'mine' if reachable else 'approach' if known else 'search'
        self.rank = {'search':0,'approach':1,'mine':2}[self.stage]
        self.blocked = None
        self.completed = False

    def observe(self, *, tick, interactions, inventory_count, known, reachable, complete=False):
        tick = count(tick, 'tick')
        interactions = count(interactions, 'interactions')
        inventory_count = count(inventory_count, 'inventory_count')
        if tick <= self.last_tick or interactions < self.interactions:
            raise ValueError('stale observation or interaction counter reset')
        if type(complete) is not bool:
            raise ValueError('invalid completion evidence')
        if self.completed or self.blocked:
            raise RuntimeError('progress after completion or blocked execution')
        if not complete and (type(known) is not bool or type(reachable) is not bool):
            raise ValueError('unknown target evidence')
        if not complete and reachable and not known:
            raise ValueError('reach requires known target')
        produced = interactions > self.interactions
        acquired = inventory_count > self.inventory_peak
        if complete:
            self.stage, self.completed = 'complete', True
        else:
            if produced and self.pending_since is None:
                self.pending_since = tick
            if acquired:
                self.pending_since = None
                self.rank = -1  # A new acquisition cycle may start.
            candidate = ('collect' if self.pending_since is not None else
                         'mine' if reachable else 'approach' if known else 'search')
            rank = {'search':0, 'approach':1, 'mine':2, 'collect':3}[candidate]
            if produced or acquired or rank > self.rank:
                self.last_progress_tick = tick
            self.rank = max(self.rank, rank)
            self.stage = candidate
            if self.pending_since is not None and tick-self.pending_since >= self.recovery_ticks:
                self.blocked = 'result_not_acquired'
            elif tick-self.last_progress_tick >= self.stall_ticks:
                self.blocked = 'no_confirmed_progress'
        self.last_tick, self.interactions = tick, interactions
        self.inventory_count = inventory_count
        self.inventory_peak = max(self.inventory_peak, inventory_count)
        return self.snapshot()

    def snapshot(self):
        return dict(stage=self.stage, completed=self.completed, blocked=self.blocked,
                    tick=self.last_tick, last_progress_tick=self.last_progress_tick,
                    pending_result_since=self.pending_since, confirmed_interactions=self.interactions,
                    inventory_count=self.inventory_count, inventory_peak=self.inventory_peak,
                    stall_ticks=self.stall_ticks, recovery_ticks=self.recovery_ticks)
