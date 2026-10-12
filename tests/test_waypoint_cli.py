import unittest
from minecraftrl import cli


class WaypointCliTests(unittest.TestCase):
    def arguments(self):
        return cli.parser().parse_args(['episode','--policy','steve1','--condition','wood',
            '--rl-collect','--rl-split','eval','--rl-reward-mode','wood-potential-v4',
            '--rl-checkpoint','results/parent.pt','--rl-embed','results/wood.npy',
            '--rl-world','results/world.zip','--rl-waypoint-actor','results/navigation.pt'])

    def test_explicit_eval_only_and_no_controller_substitution(self):
        args=self.arguments();cli.validate_waypoint_args(args)
        for key,value in [('rl_split','train'),('rl_recovery_embed','results/explore.npy'),('rl_world',None),
            ('rl_teacher',True),('rl_student_correction',True),('rl_probe_route','detour'),('rl_prepare',True),
            ('rl_checkpoint',None),('rl_embed',None),('rl_collect',False),('rl_reward_mode','sparse')]:
            with self.subTest(key=key):
                args=self.arguments();setattr(args,key,value)
                with self.assertRaises(SystemExit):cli.validate_waypoint_args(args)

    def test_initial_route_requires_explicit_waypoint_actor(self):
        args=self.arguments();args.rl_waypoint_start='initial';cli.validate_waypoint_args(args)
        args.rl_waypoint_actor=None
        with self.assertRaises(SystemExit):cli.validate_waypoint_args(args)

    def test_visit_planner_requires_waypoint_actor(self):
        args=self.arguments();args.rl_waypoint_planner='visit_memory';cli.validate_waypoint_args(args)
        args.rl_waypoint_actor=None
        with self.assertRaises(SystemExit):cli.validate_waypoint_args(args)


if __name__=='__main__':unittest.main()
