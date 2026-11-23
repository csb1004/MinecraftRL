"""Attack key from the dig head's own probability (user decision 2026-11-24, comparison arm only): above one half the
attack button is pressed, otherwise released; every other button and the camera stay as sampled."""
import sys
import unittest
from pathlib import Path

import torch

from minecraftrl.attack_decode import AttackThreshold
from minecraftrl.build_sensor_repair import DIG_STAGES
from minecraftrl.reacquire_head import ReacquireHead

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'external/steve1'))
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping  # noqa: E402

MAPPER = CameraHierarchicalMapping(n_camera_bins=11)
GROUPS = list(MAPPER.BUTTONS_GROUPS)


def combo(**pressed):
    """Joint button index with these groups set (the rest 'none')."""
    return MAPPER.BUTTONS_COMBINATION_TO_IDX[tuple(pressed.get(g, 'none') for g in GROUPS)]


def pd(probs):
    """Guided button logits [1, 1, 1, N] (not normalised, as after classifier-free guidance) for {index: probability}."""
    logits = torch.full((1, 1, 1, len(MAPPER.BUTTONS_COMBINATIONS)), -30.)
    for i, p in probs.items(): logits[..., i] = torch.log(torch.tensor(p)) + 3.
    return dict(buttons=logits, camera=torch.zeros(1, 1, 1, 121))


def sampled(index, camera=60):
    return dict(buttons=torch.tensor([[[index]]]), camera=torch.tensor([[[camera]]]))


NONE, ATTACK = combo(), combo(attack='attack')
FORWARD_CAM, FORWARD_ATTACK_CAM = combo(fore_back='forward', camera='camera'), combo(fore_back='forward', attack='attack',
                                                                                   camera='camera')


class AttackThresholdTests(unittest.TestCase):
    def setUp(self): self.decode = AttackThreshold(MAPPER)

    def test_attack_above_one_half_is_pressed_although_the_sample_missed_it(self):
        out = self.decode(pd({ATTACK: .8, NONE: .2}), sampled(NONE))
        self.assertEqual(out['buttons'].item(), ATTACK)
        self.assertEqual(self.decode.stats(), dict(rule='attack pressed iff the head gives it more than 0.5', rows=1,
                                                   pressed=1, released=0))

    def test_attack_at_or_below_one_half_is_released_although_the_sample_drew_it(self):
        out = self.decode(pd({ATTACK: .3, FORWARD_CAM: .7}), sampled(ATTACK))
        self.assertEqual(out['buttons'].item(), NONE)
        out = self.decode(pd({ATTACK: .5, NONE: .5}), sampled(ATTACK))
        self.assertEqual(out['buttons'].item(), NONE)
        self.assertEqual(self.decode.stats()['released'], 2)

    def test_the_other_buttons_and_the_camera_stay_as_sampled(self):
        out = self.decode(pd({FORWARD_ATTACK_CAM: .6, FORWARD_CAM: .4}), sampled(FORWARD_CAM, camera=17))
        self.assertEqual((out['buttons'].item(), out['camera'].item()), (FORWARD_ATTACK_CAM, 17))
        out = self.decode(pd({ATTACK: .9, NONE: .1}), sampled(ATTACK))                  # already right: unchanged
        self.assertEqual(out['buttons'].item(), ATTACK)
        self.assertEqual(self.decode.stats()['pressed'], 1)

    def test_the_inventory_button_is_left_alone(self):
        inventory = MAPPER.BUTTONS_COMBINATION_TO_IDX['inventory']
        out = self.decode(pd({ATTACK: .9, inventory: .1}), sampled(inventory))
        self.assertEqual(out['buttons'].item(), inventory)


class Base(torch.nn.Module):
    def forward(self, pi_h, mask=None): return pi_h

    def sample(self, pd, deterministic=False): return sampled(NONE)


class HeadDecodeTests(unittest.TestCase):
    def test_the_head_decodes_only_in_its_decode_stages_while_active(self):
        head = ReacquireHead(Base(), 4, use_stage=True, stages=DIG_STAGES)
        head.decode = AttackThreshold(MAPPER); head.decode_stages = ('dig',)
        likely = pd({ATTACK: .9, NONE: .1})
        head.stage = DIG_STAGES.index('dig')
        self.assertEqual(head.sample(likely, deterministic=False)['buttons'].item(), ATTACK)
        head.stage = DIG_STAGES.index('collect')
        self.assertEqual(head.sample(likely, deterministic=False)['buttons'].item(), NONE)
        head.stage = DIG_STAGES.index('dig'); head.active = False
        self.assertEqual(head.sample(likely, deterministic=False)['buttons'].item(), NONE)
        self.assertEqual(head.decode_stats(), dict(head.decode.stats(), stages=['dig']))

    def test_without_a_decode_the_head_samples_as_before(self):
        head = ReacquireHead(Base(), 4, use_stage=True, stages=DIG_STAGES); head.stage = DIG_STAGES.index('dig')
        self.assertEqual(head.sample(pd({ATTACK: .9, NONE: .1}))['buttons'].item(), NONE)
        self.assertIsNone(head.decode_stats())


if __name__ == '__main__':
    unittest.main()
