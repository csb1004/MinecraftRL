import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('craft_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_craft_recipes.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class CraftAnalysisTests(unittest.TestCase):
    def test_delayed_inherited_statistics_use_server_baseline(self):
        first={'observation':{'craft_item':{'oak_planks':0,'stick':0}},'info':{'craft_gui':{
            'server_available':1,'server_crafted_planks':4,'server_crafted_sticks':0,'server_crafted_table':0}}}
        baseline=analysis.craft_stat_baseline(first)
        self.assertEqual(baseline['oak_planks'],4)
        self.assertEqual(baseline['stick'],0)
        self.assertEqual(first['observation']['craft_item']['oak_planks'],0)
