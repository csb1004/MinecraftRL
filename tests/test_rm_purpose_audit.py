import unittest

from scripts.analyze_rm_purpose import release_check


class ReleaseAuditTests(unittest.TestCase):
    def terminal_summary(self):
        return dict(status='environment_done', release='environment_done_no_neutral_step',
                    worker_exit_code=0, process_cleanup=dict(remaining_live_pids=[]))

    def test_death_keeps_performance_sample_without_claiming_neutral_release(self):
        evidence=release_check(self.terminal_summary(), {}, True)
        self.assertTrue(evidence['valid'])
        self.assertFalse(evidence['neutral_verified'])
        self.assertIsNone(evidence['held_key_count'])

    def test_terminal_exception_requires_actual_terminal_and_complete_cleanup(self):
        self.assertFalse(release_check(self.terminal_summary(), {}, False)['valid'])
        for key,value in [('status','time_limit'),('worker_exit_code',-15),
                          ('process_cleanup',dict(remaining_live_pids=[123]))]:
            summary=self.terminal_summary()
            summary[key]=value
            self.assertFalse(release_check(summary, {}, True)['valid'])

    def test_live_environment_missing_or_nonzero_release_receipt_fails(self):
        summary=dict(status='time_limit', release='neutral_command_sent')
        self.assertFalse(release_check(summary, {}, False)['valid'])
        self.assertFalse(release_check(summary, dict(verified=True,held_key_count=1), False)['valid'])
        self.assertTrue(release_check(summary, dict(verified=True,held_key_count=0), False)['valid'])


if __name__=='__main__':
    unittest.main()
