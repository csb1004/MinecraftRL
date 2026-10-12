from copy import deepcopy
import unittest
from test_planning_recovery import Game


class SubgoalSkillTests(unittest.TestCase):
    def test_tracked_target_is_not_reachable_or_completed(self):
        from minecraftrl.subgoal_skills import navigation_state
        obs=deepcopy(Game().state)
        obs.update(wood_v3_kind=1,wood_v3_reachable=0,wood_v3_distance=8,
                   wood_v3_target_x=5,wood_v3_target_y=64,wood_v3_target_z=0)
        state=navigation_state(obs,0)
        self.assertEqual(state['phase'],'tracked_unreachable')
        self.assertTrue(state['target_known']);self.assertFalse(state['interaction_ready'])

    def test_displacement_is_horizontal_not_camera_or_vertical_motion(self):
        from minecraftrl.subgoal_skills import navigation_state,completion
        obs=deepcopy(Game().state);before=navigation_state(obs,0)
        obs['server_y']+=4;obs['yaw']=180
        self.assertFalse(completion('leave_stalled_area',before,navigation_state(obs,0)))
        obs['server_x']+=3
        self.assertTrue(completion('leave_stalled_area',before,navigation_state(obs,0)))

    def test_invalid_and_dead_observations_cannot_complete_subgoal(self):
        from minecraftrl.subgoal_skills import navigation_state,completion
        obs=deepcopy(Game().state);before=navigation_state(obs,0)
        for patch in ({'server_available':0},{'server_alive':0},{'server_x':float('nan')}):
            current=deepcopy(obs);current.update(patch)
            with self.assertRaises(ValueError):navigation_state(current,0)
        self.assertFalse(completion('approach_wood',before,before))

    def test_inventory_completion_precedes_subgoal_labels(self):
        from minecraftrl.subgoal_skills import navigation_state
        obs=deepcopy(Game().state);obs['server_inventory']['oak_log']=1
        self.assertEqual(navigation_state(obs,0)['phase'],'completed')

    def test_open_inventory_is_observed_without_claiming_navigation_completion(self):
        from minecraftrl.subgoal_skills import navigation_state,completion
        obs=deepcopy(Game().state);before=navigation_state(obs,0)
        obs.update(screen=1,wood_v3_reachable=1);obs['server_x']+=4
        state=navigation_state(obs,0)
        self.assertEqual(state['phase'],'gui_open')
        self.assertFalse(state['interaction_ready'])
        for name in ('leave_stalled_area','locate_wood','approach_wood'):
            self.assertFalse(completion(name,before,state))

    def test_registry_binds_embedding_and_completion_not_free_text(self):
        from minecraftrl.subgoal_skills import capability,validate_capability
        value=capability('approach_wood','policy','a'*64)
        self.assertEqual(validate_capability(value),value)
        for patch in ({'completion':'walked_somewhere'},{'max_budget_ticks':201}):
            changed=deepcopy(value);changed.update(patch)
            with self.assertRaises(ValueError):validate_capability(changed)
