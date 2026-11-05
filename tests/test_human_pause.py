"""Observed pauses must not become wall-contact, falling or task-success labels."""
import copy
import unittest
from minecraftrl.human_pause import observed_pause_evidence


def example():
    rows=[];actions={}
    for i in range(41):
        rows.append(dict(tick=i,isGuiOpen=False,xpos=min(i,28)*.12,ypos=64.,zpos=0.))
        actions[i]=dict(bc_eligible=True,action=dict(camera=[0.,1.],forward=int(i<28),back=0,
            left=0,right=0,jump=0,sneak=0,sprint=0,attack=0,use=0,drop=0,inventory=0))
    return rows,actions


class HumanPauseTests(unittest.TestCase):
    def test_observed_release_and_stability_without_ground_intent_or_success(self):
        raw,actions=example();proof=observed_pause_evidence(raw,actions,0,40)
        self.assertEqual(proof['pause_start'],28)
        self.assertEqual(proof['observed_transitions'],12)
        self.assertEqual(proof['maximum_horizontal_step'],0.)
        self.assertFalse(proof['ground_observed'])
        self.assertFalse(proof['human_intent_observed'])
        self.assertFalse(proof['task_completion_observed'])
        self.assertTrue(proof['movement_buttons_released'])

    def test_wall_contact_while_holding_forward_is_not_a_pause(self):
        raw,actions=example();actions[32]['action']['forward']=1
        with self.assertRaisesRegex(ValueError,'released'):observed_pause_evidence(raw,actions,0,40)

    def test_falling_drift_or_unsteady_reference_rejected(self):
        for frame,field,change in ((32,'ypos',.1),(32,'xpos',.04),(40,'xpos',.15)):
            raw,actions=example();raw[frame][field]+=change
            with self.subTest(frame=frame,field=field),self.assertRaises(ValueError):
                observed_pause_evidence(raw,actions,0,40)

    def test_idle_spawn_is_not_a_movement_then_pause_demo(self):
        raw,actions=example()
        for row in raw:row['xpos']=0.
        with self.assertRaisesRegex(ValueError,'approach'):observed_pause_evidence(raw,actions,0,40)

    def test_gui_interaction_and_invalid_actions_rejected(self):
        for change in ('gui','attack','ineligible','clipped','nonfinite'):
            raw,actions=example()
            if change=='gui':raw[30]['isGuiOpen']=True
            elif change=='attack':actions[30]['action']['attack']=1
            elif change=='ineligible':actions[30]['bc_eligible']=False
            elif change=='clipped':actions[30]['action']['camera']=[11.,0.]
            else:raw[30]['xpos']=float('nan')
            with self.subTest(change=change),self.assertRaises(ValueError):observed_pause_evidence(raw,actions,0,40)

    def test_missing_terminal_pose_and_nonconsecutive_ticks_rejected(self):
        raw,actions=example()
        with self.assertRaises(ValueError):observed_pause_evidence(raw[:-1],actions,0,40)
        raw[35]['tick']+=1
        with self.assertRaises(ValueError):observed_pause_evidence(raw,actions,0,40)
        for start,end in ((0,39),(True,41),(-1,39)):
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):
                observed_pause_evidence(raw,actions,start,end)

    def test_validation_does_not_mutate_observations_or_actions(self):
        raw,actions=example();before=copy.deepcopy((raw,actions))
        observed_pause_evidence(raw,actions,0,40)
        self.assertEqual((raw,actions),before)


if __name__=='__main__':unittest.main()
