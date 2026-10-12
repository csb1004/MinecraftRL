import unittest
from minecraftrl import rl_core


def observation(tick=0, kind=1, distance=3., reachable=1, logs=0):
    # Old locked target is occluded; new target is the log actually being mined.
    return dict(server_tick=tick, server_inventory={'oak_log': logs},
                wood_kind=1, wood_distance=7., wood_reachable=0,
                wood_v3_kind=kind, wood_v3_distance=distance, wood_v3_reachable=reachable)


class WoodTargetTests(unittest.TestCase):
    def machine(self, obs=None):
        return rl_core.SHAPED_REWARDS['wood-potential-v3'](observation() if obs is None else obs)

    def test_actual_target_overrides_occluded_legacy_target_only_in_v3(self):
        self.assertEqual(rl_core.ShapedWoodRewardV2(observation()).stage, 'approach')
        self.assertEqual(self.machine().stage, 'mine')
        self.assertEqual(self.machine().context(), [0., 0., 1., 0., 0., 3./16])

    def test_missing_new_sensor_fails_closed(self):
        obs=observation(); del obs['wood_v3_kind']
        with self.assertRaises(ValueError): self.machine(obs)

    def test_broken_interaction_target_collects_then_inventory_terminates(self):
        r=self.machine()
        self.assertEqual(r.advance(observation(1,kind=2,reachable=0))['stage'], 'collect')
        end=r.advance(observation(2,logs=1))
        self.assertEqual((end['task'],end['potential_after'],end['stage']), (1.,0.,'complete'))
        with self.assertRaises(RuntimeError): r.advance(observation(3,logs=1))

    def test_target_switch_round_trip_cannot_mint_discounted_shaping(self):
        r=self.machine(observation(reachable=0,distance=9.)); total=0.
        for i,obs in enumerate([observation(1),observation(2,reachable=0,distance=9.)]):
            total+=r.gamma**i*r.advance(obs)['shaping']
        self.assertLessEqual(total,0.)

    def test_failure_contract_is_preserved_and_accepted_by_training_validator(self):
        r=self.machine(); part=r.advance(observation(1),terminal=True)
        self.assertEqual(part['failure'],-1.)
        data=dict(eligible=True,split='train',scope='head_only_frozen_features',
                  condition={'reward_mode':'wood-potential-v3'},reward_config=r.metadata,
                  rows=[dict(old_logp=0.,value=0.,reward=part['total'],terminal=True,reward_components=part)])
        rl_core.validate_training_data(data)
        data['rows'][0]['reward_components']['failure']=0.
        with self.assertRaises(ValueError):rl_core.validate_training_data(data)

    def test_server_accessor_is_read_only_and_preserves_original_code(self):
        from minecraftrl.world_save_patch import patch_mining_observation
        original='public class PlayerInteractionManager {\n// original methods\n}'
        patched=patch_mining_observation(original)
        self.assertIn('receivedFinishDiggingPacket',patched)
        self.assertIn('delayedDestroyPos',patched)
        self.assertIn('isDestroyingBlock',patched)
        self.assertIn('destroyPos',patched)
        self.assertTrue(patched.endswith('// original methods\n}'))
        self.assertNotIn('tryHarvestBlock(',patched)
        with self.assertRaises(ValueError):patch_mining_observation('unexpected source')
