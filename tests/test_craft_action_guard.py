import unittest
from minecraftrl.crafting_probe import CraftingEnvironment, CraftEvidence


class CraftActionGuardTests(unittest.TestCase):
    def test_guard_changes_forwarded_action_and_preserves_model_proposal(self):
        class Env:
            def step(self, action):
                self.forwarded=action
                # A single pickup yields4, QUICK_MOVE consumes both recipes and yields8.
                count=8 if action['sneak'] else 4
                return {'inventory':{'stick':count},'craft_item':{'stick':0},
                        'location_stats':{'xpos':0.,'ypos':64.,'zpos':0.}},0,False,{
                    'isGuiOpen':True,'craft_gui':{'cursor_sticks':0,'grid_sticks':0}}
        env=Env()
        wrapper=CraftingEnvironment(env,None,None,None,recover_cursor=True,recipe='sticks',no_quick_move=True)
        wrapper.evidence=CraftEvidence({'inventory':{'oak_planks':4},'craft_item':{'stick':0}},'sticks')
        proposal={'sneak':[1],'attack':[1],'camera':[[0.,0.]]}
        obs,_,_,info=wrapper.step(proposal)
        self.assertEqual(proposal['sneak'],[1])
        self.assertEqual(env.forwarded['sneak'],0)
        self.assertEqual(env.forwarded['attack'],[1])
        self.assertEqual(obs['inventory']['stick'],4)
        self.assertEqual(wrapper.phase,'recovering')
        self.assertEqual(info['craft_action_guard']['proposed_action'],proposal)
        self.assertEqual(info['craft_action_guard']['forwarded_action']['sneak'],0)
        with self.assertRaisesRegex(RuntimeError,'dispatch forbidden'):wrapper.step(proposal)

    def test_guard_is_opt_in_and_does_not_change_baseline_actions(self):
        class Env:
            def step(self,action):
                self.forwarded=action
                return {'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}},0,False,{'isGuiOpen':False}
        env=Env(); wrapper=CraftingEnvironment(env,None,None,None)
        wrapper.evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
        proposal={'sneak':[1],'attack':[1]};wrapper.step(proposal)
        self.assertIs(env.forwarded,proposal)

    def test_guard_requires_recovery_and_complete_action(self):
        with self.assertRaises(ValueError):CraftingEnvironment(None,None,None,None,no_quick_move=True)
        wrapper=CraftingEnvironment(None,None,None,None,recover_cursor=True,no_quick_move=True)
        with self.assertRaisesRegex(ValueError,'sneak'):wrapper.step({'attack':1})
