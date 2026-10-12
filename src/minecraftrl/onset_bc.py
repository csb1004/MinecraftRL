"""Matched-update correction sampling and executable recovery-start metrics."""
import random

from .correction_bc import supervised_windows
from .correction_diagnostics import CommandMetrics


def correction_schedule(groups, splits, mode, seed=9605):
    if mode not in ('uniform', 'onset') or set(groups) != set(splits):
        raise ValueError('invalid correction sampling contract')
    continuation, onsets = [], []
    for name, rows in groups.items():
        if splits[name] not in ('train', 'development'):
            raise ValueError('evaluation data cannot enter a training schedule')
        if splits[name] != 'train':
            continue
        first = next((i for i, row in enumerate(rows) if row.get('loss_eligible')), None)
        if first is None or first < 7:
            raise ValueError('seven recorded context frames before onset required')
        if any(row.get('action') is not None for row in rows[:first]):
            raise ValueError('context cannot carry an action target')
        for row in rows[first:]:
            if row.get('owner') != 'teacher' or row.get('loss_eligible') is not True or not isinstance(row.get('action'), dict):
                raise ValueError('complete genuine teacher suffix required')
        onsets.append(dict(session=name, start=first-7, end=first+1, kind='onset'))
        continuation.extend(dict(session=name, start=a, end=b, kind='continuation')
                            for a, b in supervised_windows(rows))
    if not continuation or not onsets:
        raise ValueError('training demonstrations required')
    rng = random.Random(seed)
    first_pass = list(continuation)
    rng.shuffle(first_pass)
    if mode == 'uniform':
        second_pass = list(continuation)
        rng.shuffle(second_pass)
    else:
        second_pass = []
        while len(second_pass) < len(first_pass):
            cycle = list(onsets)
            rng.shuffle(cycle)
            second_pass.extend(cycle)
        second_pass = second_pass[:len(first_pass)]
    return [dict(item) for pair in zip(first_pass, second_pass) for item in pair]


class OnsetMetrics(CommandMetrics):
    def measure(self, pd, target=None):
        result = super().measure(pd, target)
        buttons, camera = pd['buttons'].flatten().exp(), pd['camera'].flatten().exp()
        # Camera activation and attack belong to the same categorical head: do
        # not multiply their marginal probabilities as if they were independent.
        released_camera = float(buttons[self.camera_on & ~self.masks['attack']].sum())
        result['p_release_attack_and_turn'] = released_camera * float(camera[self.yaw_sign != 0].sum())
        if target is not None:
            sign = int(self.yaw_sign[int(target['camera'].item())])
            if sign and self.camera_on[int(target['buttons'].item())]:
                result['p_release_attack_and_target_yaw'] = released_camera * float(camera[self.yaw_sign == sign].sum())
        return result
