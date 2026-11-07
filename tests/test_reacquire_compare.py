import unittest

from minecraftrl.reacquire_compare import arm_stats, verdict


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


if __name__ == '__main__':
    unittest.main()
