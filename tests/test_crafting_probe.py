import unittest
import tempfile
from unittest.mock import patch
from minecraftrl.crafting_probe import CraftEvidence


def obs(log=1,planks=0,crafted=0):
    return {'inventory':{'oak_log':log,'oak_planks':planks},'craft_item':{'oak_planks':crafted}}


class CraftEvidenceTests(unittest.TestCase):
    def test_completion_stops_before_next_policy_action(self):
        from minecraftrl.runtime import run_episode
        class Policy:
            calls=0
            def reset(self,goal): pass
            def act(self,screen): self.calls+=1; return {}
            def close(self): pass
        class Env:
            def seed(self,seed): pass
            def reset(self): return dict(obs(),pov=None)
            def step(self,action): return dict(obs(log=0,planks=4,crafted=4),pov=None),0,False,{}
            def release(self): return 'neutral_command_sent'
            def close(self): pass
        evidence=CraftEvidence(obs()); policy=Policy()
        def complete(value): evidence.observe(value,1); return evidence.witness is not None
        with tempfile.TemporaryDirectory() as folder:
            report=run_episode(policy,Env(),folder,seed=1,goal='craft wooden planks',
                completion_check=complete,save_screen=lambda *a:None)
        self.assertEqual(policy.calls,1)
        self.assertEqual(report['status'],'target_reached')
        self.assertTrue(report['execution_valid'])
        self.assertEqual(report['stop_reason'],'target_reached')

    def test_cli_rejects_original_world_store(self):
        from minecraftrl.cli import main
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        with patch('sys.argv',['probe','craft-probe','--policy','steve1','--world-slot','original',
                               '--world-store-root',str(root/'data/worlds')]):
            with self.assertRaisesRegex(SystemExit,'inside project results'): main()

    def test_requires_recipe_inventory_and_stat_evidence_together(self):
        evidence=CraftEvidence(obs())
        evidence.observe(obs(log=0),1)
        evidence.observe(obs(log=0,planks=4),2)
        self.assertIsNone(evidence.witness)
        evidence.observe(obs(log=0,planks=4,crafted=4),3)
        self.assertEqual(evidence.witness['step'],3)

    def test_existing_craft_stats_do_not_count_as_new_crafting(self):
        evidence=CraftEvidence(obs(crafted=12))
        evidence.observe(obs(log=0,planks=4,crafted=12),1)
        self.assertIsNone(evidence.witness)

    def test_cleanup_recovery_is_not_policy_success(self):
        evidence=CraftEvidence(obs())
        evidence.observe(obs(log=0,planks=4,crafted=4),1,cleanup=True)
        self.assertIsNone(evidence.witness)
        self.assertTrue(evidence.cleanup_witness)

    def test_rejects_wrong_fixture(self):
        for value in (obs(log=0),obs(planks=4)):
            with self.assertRaises(ValueError): CraftEvidence(value)
