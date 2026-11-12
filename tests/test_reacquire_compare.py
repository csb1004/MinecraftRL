import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from minecraftrl.reacquire_compare import arm_stats, compare, verdict


def step(near, total, angle=None):
    seconds = dict(far=total - near, closing=0., near=near, aimed=0., mining=0., collect=0., gui=0.)
    return dict(seconds=seconds, total_s=total, near_angle_s=angle or {'<15': 0., '15-45': 0., '45-90': 0., '>=90': near})


def run(success, valid=True, release='neutral_command_sent', pids=()):
    return dict(execution_valid=valid, success=success, release=release, remaining_live_pids=list(pids), status='x')


class VerdictTest(unittest.TestCase):
    OFFICIAL = dict(success=6, runs=8, reacquire_share=.37, median_s=137.)

    def test_verdict_requires_all_three_checks(self):
        self.assertTrue(verdict(self.OFFICIAL, dict(success=6, runs=8, reacquire_share=.18, median_s=120.))['better'])
        self.assertFalse(verdict(self.OFFICIAL, dict(success=5, runs=8, reacquire_share=.10, median_s=90.))['better'])
        self.assertFalse(verdict(self.OFFICIAL, dict(success=7, runs=8, reacquire_share=.19, median_s=140.))['better'])

    def test_no_successful_run_cannot_be_better(self):
        self.assertFalse(verdict(self.OFFICIAL, dict(success=0, runs=8, reacquire_share=0., median_s=None))['better'])


class ArmStatsTest(unittest.TestCase):
    def test_reacquire_share_pools_successful_and_failed_runs(self):
        stats = arm_stats([run(True), run(False)], [step(10., 20.), step(30., 40.)])
        self.assertAlmostEqual(stats['reacquire_share'], 40. / 60.)
        self.assertEqual((stats['success'], stats['runs'], stats['median_s']), (1, 2, 20.))
        self.assertEqual(stats['near_angle_s']['>=90'], 40.)

    def test_stop_contract_failures_are_listed(self):
        stats = arm_stats([run(True), run(True, release='failed: x'), run(True, pids=[42])],
                          [step(1., 2.), step(1., 2.), step(1., 2.)])
        self.assertEqual(stats['stop_contract_failures'], [1, 2])


class CompareTargetTest(unittest.TestCase):
    """Harder evaluation (design 2026-11-11): the timed step can be another block than iron_ore."""

    def test_the_step_of_the_named_target_is_timed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'bookshelf-s141421-n41'; run_dir.mkdir()
            (run_dir / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=True,
                                                                  release='neutral_command_sent',
                                                                  process_cleanup=dict(remaining_live_pids=[]))))
            report = dict(steps=[dict(step(5., 30.), target='bookshelf')])
            with mock.patch('minecraftrl.reacquire_compare.build_report', return_value=report):
                iron = compare({'sight': [run_dir]}, root=tmp)
                shelf = compare({'sight': [run_dir]}, root=tmp, target='bookshelf')
        self.assertIsNone(iron['arms']['sight']['median_s'])                  # the old default only reads iron steps
        self.assertEqual(shelf['arms']['sight']['median_s'], 30.)
        self.assertAlmostEqual(shelf['arms']['sight']['reacquire_share'], 5. / 30.)
        self.assertIn('bookshelf step', shelf['measure']); self.assertIn('iron step', iron['measure'])


if __name__ == '__main__':
    unittest.main()
