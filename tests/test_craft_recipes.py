import unittest
from minecraftrl.crafting_probe import CraftEvidence


class RecipeTests(unittest.TestCase):
    def evidence(self, recipe):
        return CraftEvidence({'inventory': {'oak_planks': 4, 'dirt': 1},
                              'craft_item': {'stick': 0, 'crafting_table': 0}}, recipe=recipe)

    def test_sticks_stop_before_statistics_but_require_leftovers_for_success(self):
        evidence = self.evidence('sticks')
        obs = {'inventory': {'oak_planks': 0, 'stick': 4}, 'craft_item': {'stick': 0}}
        self.assertTrue(evidence.ready_for_recovery(obs, {'cursor_sticks': 0, 'grid_sticks': 0}))
        evidence.observe(dict(obs, craft_item={'stick': 4}), 1, cleanup=True)
        self.assertIsNone(evidence.cleanup_witness)
        evidence.observe({'inventory': {'oak_planks': 2, 'stick': 4}, 'craft_item': {'stick': 4}}, 1, cleanup=True)
        self.assertIsNotNone(evidence.cleanup_witness)

    def test_table_cursor_is_real_material_but_preview_is_not(self):
        evidence = self.evidence('table')
        obs = {'inventory': {}, 'craft_item': {'crafting_table': 0}}
        self.assertFalse(evidence.ready_for_recovery(obs, {'cursor_table': 0, 'grid_table': 0}))
        self.assertTrue(evidence.ready_for_recovery(obs, {'cursor_table': 1, 'grid_table': 0}))
        gui = dict(server_available=1, server_inventory_table=1, server_cursor_count=0, server_crafted_table=1)
        self.assertTrue(evidence.server_confirmed(gui))
        self.assertFalse(evidence.server_confirmed(dict(gui, server_cursor_count=1)))

    def test_existing_targets_and_wrong_material_budget_are_rejected(self):
        for inventory in ({'oak_planks': 3}, {'oak_planks': 4, 'stick': 1}):
            with self.assertRaises(ValueError):
                CraftEvidence({'inventory': inventory, 'craft_item': {'stick': 0}}, recipe='sticks')

    def test_leftover_inputs_must_return_to_server_inventory_after_gui_close(self):
        evidence=self.evidence('sticks')
        gui=dict(server_available=1,server_inventory_sticks=4,server_cursor_count=0,server_crafted_sticks=4,
                 server_inventory_planks=0)
        self.assertTrue(evidence.server_confirmed(gui))
        self.assertFalse(evidence.final_server_confirmed(gui))
        self.assertTrue(evidence.final_server_confirmed(dict(gui,server_inventory_planks=2)))
