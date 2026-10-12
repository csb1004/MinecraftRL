import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch'),'audit requires trajectory torch')
class GoalAuditTests(unittest.TestCase):
    def check(self,*args):
        spec = importlib.util.spec_from_file_location('audit_goal_recovery',
            Path(__file__).resolve().parents[1]/'scripts/audit_goal_recovery_run.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.assert_reward_evidence(*args)

    def test_target_tuple_and_json_list_are_equivalent(self):
        native = {'shaping':-.05,'recovery':{'target':(235.,203.,-788.)}}
        saved = {'shaping':-.05,'recovery':{'target':[235.,203.,-788.]}}
        self.check(native,saved,native,saved)

    def test_value_or_target_change_still_fails(self):
        native = {'shaping':-.05,'recovery':{'target':(235.,203.,-788.)}}
        for changed in ({'shaping':.05,'recovery':{'target':[235.,203.,-788.]}},
                        {'shaping':-.05,'recovery':{'target':[235.,202.,-788.]}}):
            with self.assertRaises(AssertionError): self.check(native,native,native,changed)
