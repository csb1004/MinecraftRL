import json
import sys
import unittest
from pathlib import Path

from minecraftrl.world_save_patch import (build_save_jar, patch_demo_recipe_ids, patch_destroy_observation,
                                          patch_harvest_observation, patch_mining_observation, patched_sources)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'results/iron-demo-baseline-20261103/patch-hashes.json'
COMBOS = {'default': {}, 'craft': dict(craft_monitor=True),
          'craft+workshop': dict(craft_monitor=True, workshop_monitor=True),
          'craft+workshop+startup': dict(craft_monitor=True, workshop_monitor=True, startup_control=True)}
DECLARATION = 'public class PlayerInteractionManager {'
MARK_FLAG1 = '               boolean flag1 = this.player.func_234569_d_(blockstate);'
MARK_HARVEST = '                  block.harvestBlock(this.world, this.player, pos, blockstate, tileentity, itemstack1);'
IDS_LINE = ('        String[] ids={"oak_planks","birch_planks","spruce_planks","jungle_planks","acacia_planks",'
            '"dark_oak_planks",\n            "stick","crafting_table","wooden_pickaxe","wooden_axe","stone_axe","oak_slab"};')


def minecraft_dir():
    try:
        from minerl.env.malmo import InstanceManager
    except ImportError:
        return None
    path = Path(InstanceManager.MINECRAFT_DIR)
    return path if (path / 'build/libs/mcprec-6.13.jar').exists() else None


class DemoPatchTests(unittest.TestCase):
    def test_default_patch_hashes_unchanged(self):
        mc = minecraft_dir()
        if mc is None or not BASELINE.exists():
            self.skipTest('MCP-Reborn source or baseline is unavailable')
        baseline = json.loads(BASELINE.read_text())['hashes']
        for name, flags in COMBOS.items():
            with self.subTest(name=name):
                self.assertEqual(patched_sources(ROOT, mc, **flags)[2], baseline[name])

    def test_demo_sources_are_opt_in_additions(self):
        mc = minecraft_dir()
        if mc is None or not BASELINE.exists():
            self.skipTest('MCP-Reborn source or baseline is unavailable')
        baseline = json.loads(BASELINE.read_text())['hashes']
        for startup in (False, True):
            with self.subTest(startup=startup):
                _, extra, digest = patched_sources(ROOT, mc, craft_monitor=True, workshop_monitor=True,
                                                   startup_control=startup, demo_monitor=True)
                self.assertNotIn(digest, baseline.values())
                self.assertIn('DemoObservation.java', extra); self.assertIn('DemoFixture.java', extra)
                self.assertEqual(extra['EnvServer.java'].count('DemoObservation.snapshot()'), 1)
                observation = extra['WorkshopObservation.java']
                self.assertEqual(observation.count('DemoObservation.capture();'), 1)
                self.assertEqual(observation.count('StartupRandomTicks.capture();'), int(startup))
                self.assertEqual(observation.count('"stone_pickaxe"'), 1)
                interaction = extra['PlayerInteractionManager.java']
                self.assertEqual(interaction.count('minecraftRLRecordDestroy(block, itemstack1, pos, flag, flag1);'), 1)
                self.assertEqual(interaction.count('minecraftRLRecordHarvest(block, itemstack1, pos);'), 1)

    def test_fixture_reports_its_origin_height(self):
        # The pit depth reference (2026-11-05): the start block the fixture was placed around.
        fixture = (ROOT / 'src/minecraftrl/DemoFixture.java').read_text(encoding='utf-8')
        observation = (ROOT / 'src/minecraftrl/DemoObservation.java').read_text(encoding='utf-8')
        self.assertIn('originY=origin.getY();state=2;', fixture.replace('blocks=placed;', ''))
        self.assertIn('"demo_fixture_origin_y"', observation)

    def test_destroy_hook_inserted_after_harvest_permission_check(self):
        src = DECLARATION + '\n' + MARK_FLAG1 + '\n' + MARK_HARVEST + '\n}'
        out = patch_destroy_observation(src)
        self.assertEqual(out.count('minecraftRLRecordDestroy(block, itemstack1, pos, flag, flag1);'), 1)
        self.assertLess(out.index(MARK_FLAG1), out.index('minecraftRLRecordDestroy(block, itemstack1'))
        self.assertLess(out.index('minecraftRLRecordDestroy(block, itemstack1'), out.index(MARK_HARVEST))
        self.assertIn('public long minecraftRLDestroys', out)

    def test_destroy_hook_composes_with_existing_observation_patches(self):
        src = DECLARATION + '\n' + MARK_FLAG1 + '\n' + MARK_HARVEST + '\n}'
        out = patch_destroy_observation(patch_harvest_observation(patch_mining_observation(src)))
        self.assertEqual(out.count(DECLARATION), 1)
        for hook in ('minecraftRLRecordDestroy(block', 'minecraftRLRecordHarvest(block', 'minecraftRLMiningTarget()'):
            self.assertIn(hook, out)

    def test_destroy_hook_rejects_unexpected_source(self):
        for invalid in ('class X {}', DECLARATION + '\n' + MARK_HARVEST + '\n}',
                        DECLARATION + '\n' + MARK_HARVEST + '\n' + MARK_FLAG1 + '\n}',
                        DECLARATION + '\n' + MARK_FLAG1 + '\n' + MARK_FLAG1 + '\n' + MARK_HARVEST + '\n}'):
            with self.subTest(invalid=invalid[:60]), self.assertRaises(ValueError):
                patch_destroy_observation(invalid)

    def test_recipe_ids_gain_stone_pickaxe_once(self):
        out = patch_demo_recipe_ids(IDS_LINE)
        self.assertEqual(out.count('"stone_pickaxe"'), 1)
        for invalid in ('String[] ids={};', out):
            with self.assertRaises(ValueError):
                patch_demo_recipe_ids(invalid)

    def test_demo_requires_workshop_monitor(self):
        for call in (build_save_jar, patched_sources):
            with self.subTest(call=call.__name__), self.assertRaises(ValueError):
                call(ROOT, ROOT / 'missing-minecraft', craft_monitor=True, demo_monitor=True)

    def test_fixture_v2_encloses_the_arena(self):
        cfg = json.loads((ROOT / 'configs/iron-demo-fixture-v2.json').read_text(encoding='utf-8'))
        self.assertEqual(cfg['schema'], 'iron_demo_fixture_v1'); self.assertEqual(cfg['give'], [])
        world = {}
        for r in cfg['fill']:  # same order and bounds as DemoFixture.apply
            lo = [min(a, b) for a, b in zip(r['from'], r['to'])]; hi = [max(a, b) for a, b in zip(r['from'], r['to'])]
            self.assertTrue(all(abs(v) <= 16 for v in lo + hi))
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    for z in range(lo[2], hi[2] + 1): world[x, y, z] = r['block']
        for b in cfg['blocks']: world[b['dx'], b['dy'], b['dz']] = b['block']
        self.assertEqual((world[0, 0, 0], world[0, 1, 0], world[0, -1, 0]), ('air', 'air', 'grass_block'))
        self.assertEqual(sorted(k for k, v in world.items() if v == 'iron_ore'), [(-1, 1, 8), (1, 1, 8)])
        ring = {(x, z) for x in range(-6, 7) for z in (-3, 10)} | {(x, z) for x in (-6, 6) for z in range(-3, 11)}
        for x, z in ring:
            for y in range(-8, 9): self.assertEqual(world[x, y, z], 'barrier', (x, y, z))
        for x in range(-5, 6):
            for z in range(-2, 10): self.assertEqual(world[x, -9, z], 'barrier')
        config = json.loads((ROOT / 'configs/iron-demo.json').read_text(encoding='utf-8'))
        self.assertEqual(config['fixture'], 'configs/iron-demo-fixture-v2.json')

    def test_fixture_config_schema(self):
        cfg = json.loads((ROOT / 'configs/iron-demo-fixture.json').read_text(encoding='utf-8'))
        self.assertEqual(cfg['schema'], 'iron_demo_fixture_v1')
        self.assertIs(cfg['rotate_with_facing'], True)
        self.assertEqual(cfg['give'], [])
        self.assertTrue(any(b['block'] == 'iron_ore' for b in cfg['blocks']))
        for b in cfg['blocks']:
            self.assertEqual(set(b), {'dx', 'dy', 'dz', 'block'})
        for region in cfg.get('fill', []):
            self.assertEqual(set(region), {'from', 'to', 'block'})
            self.assertTrue(all(len(region[k]) == 3 for k in ('from', 'to')))

    @unittest.skipUnless(sys.platform == 'linux', 'MineRL specification is installed in WSL')
    def test_workshop_survival_demo_monitor_is_opt_in(self):
        from minecraftrl.workshop_monitor import workshop_survival
        normal, demo = workshop_survival(), workshop_survival(demo=True)
        self.assertNotIn('demo', [m.to_string() for m in normal.create_monitors()])
        monitors = {m.to_string(): m for m in demo.create_monitors()}
        self.assertIn('demo', monitors)
        for key in ('demo_destroys', 'demo_last_destroy_harvestable', 'demo_mining_block', 'demo_hit_block',
                    'demo_fixture_state', 'demo_fixture_sha52'):
            self.assertIn(key, monitors['demo'].space.spaces)
        self.assertIn('stone_pickaxe', monitors['workshop'].space.spaces['recipes'].spaces)
        normal_workshop = [m for m in normal.create_monitors() if m.to_string() == 'workshop'][0]
        self.assertNotIn('stone_pickaxe', normal_workshop.space.spaces['recipes'].spaces)

    def test_recipe_catalog_extra_ids_from_game_jar(self):
        mc = minecraft_dir()
        if mc is None:
            self.skipTest('MCP-Reborn jar is unavailable')
        from minecraftrl.workshop_recipes import RecipeCatalog
        jar = mc / 'build/libs/mcprec-6.13.jar'
        self.assertNotIn('stone_pickaxe', RecipeCatalog.from_jar(jar).by_output)
        catalog = RecipeCatalog.from_jar(jar, extra_ids=('stone_pickaxe',))
        steps, _ = catalog.plan({'cobblestone': 3, 'stick': 2}, ['stone_pickaxe'], workstation_available=True)
        self.assertEqual(steps, ['stone_pickaxe'])


if __name__ == '__main__':
    unittest.main()
