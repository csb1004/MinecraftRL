"""Two stage heads on one frozen action head (sensor repair 2026-11-23): the stage name picks the residual."""
import unittest

import torch

from minecraftrl.build_recover import BUILD2_STAGES
from minecraftrl.build_sensor_repair import COMBINED_STAGES, DIG_STAGES
from minecraftrl.head_switch import SwitchHead
from minecraftrl.reacquire_head import ReacquireHead


class Base(torch.nn.Module):
    def forward(self, pi_h, mask=None): return pi_h * 1.


def heads(width=4):
    base = Base()
    build = ReacquireHead(base, width, use_stage=True, stages=BUILD2_STAGES, features=tuple('s_%d' % i for i in range(3)))
    dig = ReacquireHead(base, width, use_stage=True, stages=DIG_STAGES, features=tuple('d_%d' % i for i in range(3)))
    torch.nn.init.constant_(dig.adapter[-1].bias, 1.)          # the dig residual adds 1 everywhere
    return build, dig


class SwitchHeadTests(unittest.TestCase):
    def test_the_stage_name_picks_the_head_and_its_own_stage_index(self):
        build, dig = heads(); switch = SwitchHead(build, dig)
        self.assertEqual(switch.stages, COMBINED_STAGES)
        latent = torch.zeros(1, 1, 4)
        switch.stage = COMBINED_STAGES.index('aim'); switch.feature = [0., 0., 0.]
        self.assertEqual((build.stage, dig.stage), (BUILD2_STAGES.index('aim'), None))
        self.assertTrue(torch.equal(switch(latent), latent))            # the untrained building residual is zero
        switch.stage = COMBINED_STAGES.index('dig_dig'); switch.feature = [.1, .2, .3]
        self.assertEqual((build.stage, dig.stage, dig.feature), (None, DIG_STAGES.index('dig'), [.1, .2, .3]))
        self.assertTrue(torch.equal(switch(latent), latent + 1.))
        self.assertIs(switch.last_latent, dig.last_latent)
        switch.stage = None
        self.assertEqual((build.stage, dig.stage), (None, None))

    def test_activation_reaches_both_heads_and_the_feature_counts_must_match(self):
        build, dig = heads(); switch = SwitchHead(build, dig)
        switch.active = False
        self.assertEqual((build.active, dig.active, switch.active), (False, False, False))
        switch.active = True
        self.assertTrue(build.active and dig.active)
        self.assertEqual((switch.width, switch.features, switch.use_stage), (4, build.features, True))
        odd = ReacquireHead(Base(), 4, use_stage=True, stages=DIG_STAGES, features=('d_0',))
        with self.assertRaises(ValueError): SwitchHead(build, odd)


if __name__ == '__main__':
    unittest.main()
