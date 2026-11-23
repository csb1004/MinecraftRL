"""Attack key from the dig head's own probability (user decision 2026-11-24: comparison arm only, the verdict samples).

The STEVE-1 action head samples one joint button combination per row. A block breaks only while attack stays pressed
(the beacon: 90 rows in a row), so a head that gives attack 0.85 on each row almost never holds it that long. This
decode keeps the sampled action and sets only its attack button: pressed when the head's probability of the
combinations with attack (after guidance) is above `threshold`, released otherwise. Every other button and the camera
stay as sampled, and the inventory combination is left alone. It decides from the head's own output; it adds no input
and no rule of the task. The counts of changed rows are reported with the run.
"""
import torch


class AttackThreshold:
    def __init__(self, mapper, threshold=.5):
        combos = mapper.BUTTONS_IDX_TO_COMBINATION; groups = list(mapper.BUTTONS_GROUPS); a = groups.index('attack')
        n = len(combos); self.threshold = float(threshold)
        self.mask = torch.zeros(n, dtype=torch.bool); self.toggle = torch.arange(n); self.fixed = torch.zeros(n, dtype=torch.bool)
        for i, c in combos.items():
            if not isinstance(c, tuple): self.fixed[i] = True; continue
            on = c[a] == 'attack'; self.mask[i] = on
            self.toggle[i] = mapper.BUTTONS_COMBINATION_TO_IDX[c[:a] + ('none' if on else 'attack',) + c[a + 1:]]
        self.rows = self.pressed = self.released = 0

    def __call__(self, pd, action):
        logits = pd['buttons'].float()
        probs = torch.softmax(logits, -1)
        p_attack = (probs * self.mask.to(probs.device)).sum(-1)
        index = action['buttons']; idx = index.long()
        want = (p_attack > self.threshold).reshape(idx.shape)
        has = self.mask.to(idx.device)[idx]
        flip = (has != want) & ~self.fixed.to(idx.device)[idx]
        self.rows += int(idx.numel())
        self.pressed += int((flip & want).sum()); self.released += int((flip & ~want).sum())
        out = torch.where(flip, self.toggle.to(idx.device)[idx], idx).to(index.dtype)
        return dict(action, buttons=out)

    def stats(self):
        return dict(rule='attack pressed iff the head gives it more than %g' % self.threshold, rows=self.rows,
                    pressed=self.pressed, released=self.released)
