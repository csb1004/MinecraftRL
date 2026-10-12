import copy
import unittest
from minecraftrl.bc_core import validate_plan, loss_eligible, comparison_gate

def plan():
    return dict(goal='find a cave',frame_offset=0,rm_context='unobserved_frozen',
        sessions=[dict(id='a',split='train',context_start=40,start=100,end=300),
                  dict(id='b',split='development',context_start=40,start=100,end=300)])

class BCCoreTests(unittest.TestCase):
    def test_teacher_plan_keeps_episode_start_and_observed_rm(self):
        p=plan();p.update(data_source='local_teacher',rm_context='recorded')
        for s in p['sessions']:s.update(context_start=0,start=0)
        validate_plan(p)
        p['sessions'][0]['context_start']=1
        with self.assertRaises(ValueError):validate_plan(p)
        p['sessions'][0]['context_start']=0;p.pop('data_source')
        with self.assertRaises(ValueError):validate_plan(p)

    def test_recording_chunks_from_same_contractor_session_cannot_cross_splits(self):
        p=plan()
        p['sessions'][0]['id']='cheeky-cornflower-setter-054834d1b04a-20220419-154302'
        p['sessions'][1]['id']='cheeky-cornflower-setter-054834d1b04a-20220419-154812'
        with self.assertRaises(ValueError):validate_plan(p)
        extra=dict(p['sessions'][1],split='train')
        p['sessions'][1]['id']='another-session'
        p['sessions'].append(extra)
        validate_plan(p)

    def test_split_is_by_session_and_never_eval(self):
        validate_plan(plan())
        for change in ('duplicate','evaluation'):
            p=plan()
            if change=='duplicate':p['sessions'][1]['id']='a'
            else:p['sessions'][1]['split']='eval'
            with self.assertRaises(ValueError):validate_plan(p)

    def test_context_precedes_labels_and_unknown_rm_is_not_search(self):
        for change in ('warmup','rm','offset'):
            p=plan()
            if change=='warmup':p['sessions'][0]['context_start']=101
            elif change=='rm':p['rm_context']='search'
            else:p['frame_offset']=None
            with self.assertRaises(ValueError):validate_plan(p)

    def test_masks_do_not_change_records(self):
        row=dict(bc_eligible=True,gui=False,unknown_keys=[],unsupported_actions=[],
                 action=dict(camera=[.15,-.3]))
        original=copy.deepcopy(row)
        self.assertTrue(loss_eligible(row,button_loss=False))
        self.assertFalse(loss_eligible(row,button_loss=True))
        self.assertEqual(row,original)
        row['action']['camera']=[11.,0.]
        self.assertFalse(loss_eligible(row,button_loss=False))

    def test_null_actions_can_be_kept_and_gui_excluded(self):
        row=dict(bc_eligible=True,gui=False,action=dict(camera=[0.,0.]),null_action=True)
        self.assertTrue(loss_eligible(row,button_loss=False))
        row['gui']=True
        self.assertFalse(loss_eligible(row,button_loss=False))

    def test_offline_gate_does_not_mean_policy_success(self):
        self.assertTrue(comparison_gate(5.,4.,.01,.001))
        for vals in ((5.,6.,.01,.001),(5.,4.,.04,.001),(5.,4.,.01,0.),(5.,float('nan'),.01,.001)):
            self.assertFalse(comparison_gate(*vals))
