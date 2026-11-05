"""Irreversible-action guard: decide at the last reversible moment, only for blocks the current goal needs.

Inputs are usable `harvest_requires` knowledge and the held item only; server harvestability is never read here.
"""
import numpy as np

from .planner import block_family, harvest_verdict


VIOLATION_GAP_S = 1.  # control steps are ~0.1 s apart; a longer pause means the policy looked elsewhere


def attacking(action): return bool(np.asarray(action.get('attack', 0)).any())


class Guard:
    def __init__(self, kb, context, guarded, *, violation_hold_s=10.):
        self.kb = kb; self.context = dict(context); self.guarded = frozenset(guarded); self.violation_hold_s = violation_hold_s
        self.mining_since = {}; self.violation_since = None

    def set_guarded(self, guarded): self.guarded = frozenset(guarded)

    def reset(self):
        """A (re)started step gets a fresh violation timer and mining history."""
        self.violation_since = None; self.mining_since.clear()

    def judge(self, s, block, pos, phase, mining_ticks):
        family = block_family(block)
        if family not in self.guarded: return None
        claims = self.kb.query(family, 'harvest_requires', context=self.context)
        verdict, required = harvest_verdict(claims, s.selected_item)
        base = dict(phase=phase, block=block, family=family, pos=pos, held=s.selected_item, required=required,
                    claim_ids=tuple(c.id for c in claims), mining_ticks=mining_ticks, tick=s.tick)
        if verdict == 'ok': self.violation_since = None; return None
        if verdict == 'unknown': return dict(base, kind='guard_unknown', hold=True, mask_attack=False)
        # Persistence means continuous fixation on the same block: a gap longer than VIOLATION_GAP_S restarts the timer.
        start, last = (self.violation_since[1], self.violation_since[2]) if self.violation_since else (None, None)
        if self.violation_since is None or self.violation_since[0] != pos or s.t - last > VIOLATION_GAP_S: start = s.t
        self.violation_since = (pos, start, s.t)
        persistent = s.t - start >= self.violation_hold_s
        return dict(base, kind='guard_violation_persistent' if persistent else 'guard_violation', hold=persistent, mask_attack=True)

    def pre_dispatch(self, s, action):
        """Before the action reaches the game: the aimed block from the last observation plus this action's attack."""
        if not s.valid or s.gui_open or not attacking(action) or s.hit_block in (None, 'air'): return None
        return self.judge(s, s.hit_block, s.hit_pos, 'pre_dispatch', 0)

    def observe(self, s):
        """After dispatch: the server's mining target backs up a pre-dispatch miss (e.g. the view turned in the same action)."""
        if not s.valid or s.gui_open or s.mining_pos is None or s.mining_block in (None, 'air'):
            self.mining_since.clear(); return None
        since = self.mining_since.setdefault(s.mining_pos, s.tick)
        return self.judge(s, s.mining_block, s.mining_pos, 'server_mining', s.tick - since)
