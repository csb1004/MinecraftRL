import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from minecraftrl.crafting_probe import CraftEvidence, CraftingEnvironment


def observation(planks=0, crafted=4):
    return {'inventory': {'oak_log': 0, 'oak_planks': planks},
            'craft_item': {'oak_planks': crafted}, 'pov': None,
            'location_stats':{'xpos':0.,'ypos':64.,'zpos':0.}}


class CraftRecoveryTests(unittest.TestCase):
    def test_client_prediction_does_not_satisfy_authoritative_completion(self):
        evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
        gui=dict(server_available=1,server_inventory_planks=0,server_cursor_count=4,server_crafted_planks=4)
        self.assertFalse(evidence.server_confirmed(gui))
        self.assertTrue(evidence.server_confirmed(dict(gui,server_inventory_planks=4,server_cursor_count=0)))
        self.assertFalse(evidence.server_confirmed(dict(gui,server_available=0)))
    def test_inventory_satisfaction_does_not_wait_for_delayed_craft_statistics(self):
        from minecraftrl.runtime import run_episode
        class Policy:
            calls=0
            def reset(self,goal): pass
            def act(self,screen): self.calls+=1; return {}
            def close(self): pass
        class Env:
            def seed(self,seed): pass
            def reset(self):return dict(observation(crafted=0),inventory={'oak_log':1})
            def step(self,action):return observation(planks=4,crafted=0),0,False,{}
            def release(self):return 'neutral_command_sent'
            def close(self):pass
        evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
        policy=Policy()
        with tempfile.TemporaryDirectory() as folder:
            report=run_episode(policy,Env(),folder,seed=1,goal='craft wooden planks',save_screen=lambda *a:None,
                completion_check=lambda o:evidence.ready_for_recovery(o,{'cursor_planks':0,'grid_planks':0}),
                completion_status='craft_ready_for_recovery')
        self.assertEqual(policy.calls,1)
        self.assertEqual(report['status'],'craft_ready_for_recovery')
        self.assertIsNone(evidence.witness)

    def test_recovery_phase_rejects_any_further_policy_dispatch(self):
        wrapped=CraftingEnvironment(None,None,None,None,recover_cursor=True)
        wrapped.phase='recovering'
        with self.assertRaisesRegex(RuntimeError,'dispatch forbidden'): wrapped.step({})

    def test_cursor_material_stops_policy_without_claiming_inventory_success(self):
        evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
        gui={'cursor_planks':4,'grid_planks':0}
        evidence.observe(observation(crafted=0),1)
        self.assertTrue(evidence.ready_for_recovery(observation(crafted=0),gui))
        self.assertIsNone(evidence.witness)
        self.assertFalse(evidence.ready_for_recovery(observation(crafted=0),{'cursor_planks':0,'grid_planks':0}))

    def test_cursor_is_placed_before_closing_and_cleanup_is_not_policy_success(self):
        class Raw:
            class action_space:
                @staticmethod
                def noop(): return {'camera':[0.,0.],'attack':0,'inventory':0}
        class Env:
            raw=Raw(); terminal=False
            last_observation=observation()
            actions=[]; opened=True
            gui={'inventory_screen':1,'cursor_count':4,'cursor_planks':4,'grid_planks':0,
                 'empty_slot':9,'mouse_x':80.,'mouse_y':20.,'target_x':40.,'target_y':60.,
                 'server_available':1,'server_inventory_planks':4,'server_cursor_count':0,'server_crafted_planks':4}
            def step(self,action):
                self.actions.append(action)
                self.gui=dict(self.gui,mouse_y=self.gui['mouse_y']+action['camera'][0]/.15,
                              mouse_x=self.gui['mouse_x']+action['camera'][1]/.15)
                if action['inventory']:self.opened=False
                if action['attack']:
                    self.gui=dict(self.gui,cursor_count=0,cursor_planks=0)
                    self.last_observation=observation(4)
                return self.last_observation,0,False,{'isGuiOpen':self.opened,'craft_gui':self.gui}
        env=Env()
        with tempfile.TemporaryDirectory() as folder, patch('minecraftrl.crafting_probe.save_png'):
            wrapped=CraftingEnvironment(env,None,None,Path(folder),recover_cursor=True)
            wrapped.evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
            wrapped.gui_open=True; wrapped.gui=env.gui
            wrapped.release()
        click=next(i for i,a in enumerate(env.actions) if a['attack'])
        close=next(i for i,a in enumerate(env.actions) if a['inventory'])
        self.assertLess(click,close)
        self.assertEqual(env.actions[1]['camera'],[6.,-6.])
        self.assertIsNone(wrapped.evidence.witness)
        self.assertIsNotNone(wrapped.evidence.cleanup_witness)
        self.assertTrue(wrapped.cleanup['cursor_verified_empty'])

    def test_missing_gui_telemetry_fails_closed(self):
        evidence=CraftEvidence({'inventory':{'oak_log':1},'craft_item':{'oak_planks':0}})
        with self.assertRaises(ValueError): evidence.ready_for_recovery(observation(),None)
