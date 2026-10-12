import unittest
from minecraftrl.rl_expanded import validate_rollout, training_paths


class ExpandedBatchTests(unittest.TestCase):
    def report(self):
        return dict(execution_valid=True, status='time_limit', release='neutral_command_sent',
                    process_cleanup={'remaining_live_pids': []}, rl_split='train',
                    steps=2, simulation_ticks=2, server_tick_deltas=[1, 1], rl_updates=0,
                    condition={'recurrent_commit_mode': 'dispatch_commit'})

    def test_valid_timeout_is_retained_and_cancellation_is_rejected(self):
        report = self.report()
        validate_rollout(report, 'train')
        report['status'] = 'cancelled'
        with self.assertRaises(ValueError):
            validate_rollout(report, 'train')

    def test_tick_gap_or_live_child_blocks_training(self):
        report = self.report(); report['server_tick_deltas'] = [1, 2]
        with self.assertRaises(ValueError): validate_rollout(report, 'train')
        report = self.report(); report['process_cleanup']['remaining_live_pids'] = [12]
        with self.assertRaises(ValueError): validate_rollout(report, 'train')

    def test_missing_duplicate_or_evaluation_trajectory_is_rejected(self):
        runs = [dict(label='train-1-1', path='a', report=self.report()),
                dict(label='train-1-2', path='b', report=self.report())]
        self.assertEqual(len(training_paths(runs, [1])), 2)
        with self.assertRaises(ValueError): training_paths(runs[:1], [1])
        runs[1]['path'] = 'a'
        with self.assertRaises(ValueError): training_paths(runs, [1])
        runs[1]['path'] = 'b'; runs[1]['report']['rl_split'] = 'eval'
        with self.assertRaises(ValueError): training_paths(runs, [1])
