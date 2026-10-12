import unittest
from minecraftrl.rl_validation import summarize_comparison

class ValidationComparisonTests(unittest.TestCase):
    def rows(self):
        return [dict(seed=s,condition=c,repeat=r,report=dict(status='target_reached',execution_valid=True,
            server_target_reached=True,elapsed_control_s=80 if c=='candidate' else 100,steps=100))
            for s in range(5) for c in ('parent','control','candidate') for r in (1,2)]
    def test_complete_improvement_passes(self):
        result=summarize_comparison(self.rows(),list(range(5)))
        self.assertTrue(result['candidate_gate_passed']);self.assertEqual(result['worlds_faster_than_both'],5)
    def test_missing_run_cannot_pass(self):
        result=summarize_comparison(self.rows()[:-1],list(range(5)))
        self.assertFalse(result['complete']);self.assertFalse(result['candidate_gate_passed'])
    def test_fast_failure_is_penalized_and_success_regression_blocks(self):
        rows=self.rows();row=next(x for x in rows if x['condition']=='candidate')
        row['report'].update(status='environment_done',server_target_reached=False,elapsed_control_s=1)
        result=summarize_comparison(rows,list(range(5)))
        self.assertFalse(result['candidate_gate_passed'])
        self.assertEqual(result['conditions']['candidate']['worlds']['0']['restricted_mean_s'],130)
    def test_technical_failure_excluded_not_counted_as_game_failure(self):
        rows=self.rows();rows[0]['report'].update(execution_valid=False,status='worker_failed')
        result=summarize_comparison(rows,list(range(5)))
        self.assertFalse(result['complete']);self.assertEqual(result['conditions']['parent']['valid'],9)
        self.assertEqual(result['conditions']['parent']['successes'],9)
    def test_duplicate_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):summarize_comparison(rows+[rows[0]],list(range(5)))
