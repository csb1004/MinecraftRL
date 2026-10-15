import unittest
from pathlib import Path
from unittest.mock import patch

from minecraftrl import cli


class GoalRecoveryCliTests(unittest.TestCase):
    def arguments(self):
        argv = ['episode','--policy','steve1','--condition','wood','--rl-collect','--rl-split','eval',
                '--rl-reward-mode','wood-potential-v4','--rl-checkpoint','results/parent.pt',
                '--rl-embed','results/wood.npy','--rl-world','results/world.zip',
                '--rl-recovery-embed','results/explore.npy']
        try:
            return cli.parser().parse_args(argv)
        except SystemExit:
            self.fail('explicit goal-recovery experiment cannot be selected')

    def test_explicit_eval_route_accepts_project_local_embedding(self):
        args = self.arguments()
        self.assertEqual(args.rl_recovery_embed,Path('results/explore.npy'))
        self.assertTrue(hasattr(cli,'validate_goal_recovery_args'))
        cli.validate_goal_recovery_args(args)

    def test_training_and_other_controllers_cannot_enable_recovery_probe(self):
        for key,value in [('rl_split','train'),('rl_teacher',True),('rl_student_correction',True),
                          ('rl_probe_route','detour'),('rl_prepare',True),('rl_collect',False),
                          ('rl_checkpoint',None),('rl_embed',None),('rl_world',None),('rl_reward_mode','sparse')]:
            with self.subTest(key=key):
                args = self.arguments()
                setattr(args,key,value)
                self.assertTrue(hasattr(cli,'validate_goal_recovery_args'))
                with self.assertRaisesRegex(SystemExit,'eval'):
                    cli.validate_goal_recovery_args(args)

    def test_embedding_outside_results_is_rejected_before_startup(self):
        args = self.arguments()
        args.rl_recovery_embed = args.root.parent/'outside.npy'
        self.assertTrue(hasattr(cli,'validate_goal_recovery_args'))
        with self.assertRaisesRegex(SystemExit,'project-local'):
            cli.validate_goal_recovery_args(args)

    def test_fixed_planner_requires_recovery_and_finite_delay(self):
        args=self.arguments();args.rl_planner_choice='explore_then_resume'
        cli.validate_goal_recovery_args(args)
        for delay in (-1,float('nan'),61):
            args.rl_planner_delay=delay
            with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)
        args.rl_planner_delay=5;args.rl_recovery_embed=None
        with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)

    def test_mailbox_requires_isolated_eval_and_explicit_live_mode(self):
        args=self.arguments();args.rl_planner_exchange=args.root/'results/new-mailbox'
        cli.validate_goal_recovery_args(args)
        for key,value in [('rl_planner_choice','explore_then_resume'),('rl_recovery_embed',None),
                          ('rl_planner_exchange',args.root.parent/'outside')]:
            with self.subTest(key=key):
                changed=self.arguments();changed.rl_planner_exchange=args.rl_planner_exchange
                setattr(changed,key,value)
                with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(changed)
        args=self.arguments();args.rl_planner_exchange_live=True
        with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)
        parsed=cli.parser().parse_args(['episode','--policy','steve1','--rl-planner-exchange','results/mailbox',
                                       '--rl-planner-exchange-live'])
        self.assertTrue(parsed.rl_planner_exchange_live)

    def test_rm_plan_protocol_is_explicit_and_has_bounded_requests(self):
        args=self.arguments();args.rl_rm_plan_fixture='explore';args.rl_planner_max_requests=3
        cli.validate_goal_recovery_args(args)
        args.rl_split='train'
        with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)
        args=self.arguments();args.rl_planner_max_requests=2
        with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)
        args=self.arguments();args.rl_rm_plan=True
        with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(args)
        args.rl_planner_exchange=args.root/'results/new-rm-mailbox'
        cli.validate_goal_recovery_args(args)

    def test_rm_navigation_requires_bound_subgoal_planning_and_project_weights(self):
        def args():
            value=self.arguments();value.rl_rm_plan_fixture='subgoals'
            value.rl_skill_catalogue=Path('results/catalogue.json')
            value.rl_rm_navigation_actor=Path('results/navigation.pt')
            return value
        with patch('minecraftrl.subgoal_skills.load_catalogue'):
            cli.validate_goal_recovery_args(args())
            for key,value in [('rl_skill_catalogue',None),('rl_rm_plan_fixture','explore'),
                              ('rl_waypoint_actor',Path('results/other.pt')),
                              ('rl_rm_navigation_actor',Path('../outside.pt')),
                              ('rl_recovery_embed',None),('rl_split','train')]:
                with self.subTest(key=key):
                    changed=args();setattr(changed,key,value)
                    with self.assertRaises(SystemExit):cli.validate_goal_recovery_args(changed)


if __name__ == '__main__':
    unittest.main()
