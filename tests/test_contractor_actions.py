import copy
import unittest
from minecraftrl.contractor_actions import ContractorActions


def record(**updates):
    row=dict(mouse=dict(dx=1,dy=-2,buttons=[],newButtons=[]),keyboard=dict(keys=[]),
             hotbar=0,isGuiOpen=False,tick=0)
    row.update(updates)
    return row


class ContractorActionTests(unittest.TestCase):
    def test_fractional_camera_and_source_immutability(self):
        row=record();original=copy.deepcopy(row)
        out=ContractorActions().convert(row)
        self.assertEqual(out['action']['camera'],[-.3,.15])
        self.assertEqual(row,original)

    def test_hotbar_change_is_not_null(self):
        c=ContractorActions();c.convert(record())
        out=c.convert(record(tick=1,hotbar=3,mouse=dict(dx=0,dy=0,buttons=[],newButtons=[])))
        self.assertEqual(out['action']['hotbar.4'],1)
        self.assertFalse(out['null_action'])

    def test_gui_and_unknown_keys_are_explicitly_ineligible(self):
        for row in (record(isGuiOpen=True),record(keyboard=dict(keys=['key.keyboard.f3']))):
            self.assertFalse(ContractorActions().convert(row)['bc_eligible'])

    def test_stuck_attack_heuristic_and_release(self):
        c=ContractorActions()
        first=record(mouse=dict(dx=0,dy=0,buttons=[0],newButtons=[0]))
        self.assertEqual(c.convert(first)['action']['attack'],0)
        first['tick']=1
        self.assertEqual(c.convert(first)['action']['attack'],1)

    def test_bad_values_and_temporal_gap_are_rejected(self):
        for updates in (dict(hotbar=9),dict(mouse=dict(dx=float('nan'),dy=0,buttons=[],newButtons=[]))):
            with self.assertRaises(ValueError):ContractorActions().convert(record(**updates))
        c=ContractorActions();c.convert(record())
        with self.assertRaises(ValueError):c.convert(record(tick=2))

    def test_unsupported_actions_are_not_silently_learned(self):
        out=ContractorActions().convert(record(keyboard=dict(keys=['key.keyboard.f'])))
        self.assertFalse(out['bc_eligible'])
        self.assertIn('swapHands',out['unsupported_actions'])
