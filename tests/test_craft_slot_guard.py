import unittest
from minecraftrl.crafting_probe import guard_result_slot


class CraftSlotGuardTests(unittest.TestCase):
    def gui(self,**values):
        return dict(dict(inventory_screen=1,mouse_x=40.,mouse_y=50.,result_x=100.,result_y=50.,
                         gui_width=640.,gui_height=360.),**values)
    def action(self,**values):
        return dict(dict(sneak=[1],attack=[1],inventory=[0],camera=[[0.,0.]]),**values)
    def test_move_and_click_on_result_releases_shift_before_forwarding(self):
        proposal=self.action(camera=[[0.,9.]])
        forwarded,audit=guard_result_slot(proposal,self.gui())
        self.assertEqual(forwarded['sneak'],0)
        self.assertEqual(proposal['sneak'],[1])
        self.assertEqual(forwarded['camera'],proposal['camera'])
        self.assertEqual(audit['predicted_mouse'],[100.,50.])
        self.assertEqual(audit['reason'],'result_slot')
    def test_material_slot_and_world_keep_shift(self):
        for gui in (self.gui(),dict(inventory_screen=0)):
            proposal=self.action()
            forwarded,_=guard_result_slot(proposal,gui)
            self.assertEqual(forwarded,proposal)
    def test_gui_transition_and_bootstrap_release_shift(self):
        for gui,action in ((None,self.action()),(self.gui(),self.action(inventory=[1]))):
            forwarded,_=guard_result_slot(action,gui)
            self.assertEqual(forwarded['sneak'],0)
    def test_exact_slot_bounds_and_clipped_camera(self):
        self.assertEqual(guard_result_slot(self.action(camera=[[0.,7.65]]),self.gui())[0]['sneak'],0)
        self.assertEqual(guard_result_slot(self.action(camera=[[0.,10.35]]),self.gui())[0]['sneak'],[1])
        self.assertEqual(guard_result_slot(self.action(camera=[[0.,180.]]),self.gui())[1]['predicted_mouse'],[640.,50.])
    def test_missing_geometry_fails_closed(self):
        with self.assertRaises(ValueError):guard_result_slot(self.action(),{'inventory_screen':1})
    def test_unrecognized_gui_is_not_treated_as_world(self):
        with self.assertRaisesRegex(ValueError,'unsupported GUI'):
            guard_result_slot(self.action(),{'inventory_screen':0,'other_screen_open':True})
