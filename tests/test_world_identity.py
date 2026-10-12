import unittest
from unittest.mock import patch


class WorldIdentityTests(unittest.TestCase):
    def test_fresh_path_keeps_original_game_rng_call_and_rejects_collision(self):
        from minecraftrl.world_save_patch import patch_world_identity
        marker = 'String worldName = "mcpworld" + RandomHelper.getRandomHexString();'
        source = marker + '\nmc.createWorld(worldName, worldSettings, registries, dms);'
        result = patch_world_identity(source)
        self.assertEqual(result.count('RandomHelper.getRandomHexString()'), 1)
        self.assertIn('java.util.UUID.randomUUID()', result)
        self.assertIn('java.nio.file.Files.exists', result)
        self.assertLess(result.index('throw new IllegalStateException'), result.index('mc.createWorld'))
        with self.assertRaises(ValueError): patch_world_identity('changed upstream')

    def test_saved_seed_must_match_request(self):
        from minecraftrl.rl_worlds import verify_saved_seed
        with patch('minecraftrl.saved_inventory.read_nbt', return_value={'Data': {'WorldGenSettings': {'seed': 12}}}):
            self.assertEqual(verify_saved_seed(b'unused', 12), 12)
            with self.assertRaises(ValueError): verify_saved_seed(b'unused', 13)


if __name__ == '__main__': unittest.main()
