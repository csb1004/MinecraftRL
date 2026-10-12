import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from minecraftrl.cli import main
from minecraftrl.reporting import aggregate

class ReportingTests(unittest.TestCase):
    def test_planning_pause_is_separate_from_technical_failure_and_complete_trial(self):
        report=aggregate([dict(status='planning_paused',execution_valid=False,target_observed=False)],
                         planned=2,has_target=True)
        self.assertEqual(report['planning_paused'],1)
        self.assertEqual(report['technical_failures'],0)
        self.assertEqual(report['unrun'],1)
        self.assertEqual(report['valid_gameplay_episodes'],0)
        self.assertIsNone(report['task_failure_rate'])

    def test_acknowledged_clean_session_cancel_is_not_cli_failure(self):
        report={'status':'cancelled','execution_valid':False,'task':{'status':'cancelled'},
                'release':'neutral_command_sent','worker_exit_code':2,
                'process_cleanup':{'remaining_live_pids':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'conditions':[
                {'policy':'steve1','name':'wood','goal':'collect wood','target_items':['oak_log']}]}))
            argv=['probe','session','--policy','steve1','--root',str(root)]
            with patch('sys.argv',argv),patch('minecraftrl.cli.episode',return_value=report),patch('builtins.print'):
                self.assertEqual(main(),0)
            failed=dict(report,environment_close_error='close failed')
            with patch('sys.argv',argv),patch('minecraftrl.cli.episode',return_value=failed),patch('builtins.print'):
                self.assertEqual(main(),2)
            forced=dict(report,worker_exit_code=-9)
            with patch('sys.argv',argv),patch('minecraftrl.cli.episode',return_value=forced),patch('builtins.print'):
                self.assertEqual(main(),2)
            failed_save=dict(report,world_save={'status':'failed','error':'disk full'})
            with patch('sys.argv',argv),patch('minecraftrl.cli.episode',return_value=failed_save),patch('builtins.print'):
                self.assertEqual(main(),2)

    def test_unrun_and_blocked_are_not_task_failures(self):
        report = aggregate([{'status': 'blocked', 'execution_valid': False},
                            {'status': 'environment_done', 'execution_valid': True, 'target_observed': True},
                            {'status': 'time_limit', 'execution_valid': True, 'target_observed': False}], planned=5, has_target=True)
        self.assertEqual(report['unrun'], 2)
        self.assertEqual(report['valid_gameplay_episodes'], 2)
        self.assertEqual(report['task_failure_rate'], .5)
        self.assertEqual(report['technical_failure_rate'], 1 / 3)

    def test_foundation_has_no_instruction_success_rate(self):
        self.assertIsNone(aggregate([{'status':'time_limit','execution_valid':True}], planned=5, has_target=False)['task_failure_rate'])

    def test_empty_sample_has_null_rates(self):
        result = aggregate([], planned=5, has_target=True)
        self.assertIsNone(result['technical_failure_rate'])
        self.assertIsNone(result['task_failure_rate'])

    def test_blocked_suite_keeps_other_condition_as_unrun(self):
        self.check_invalid_suite('blocked')

    def test_runtime_error_stops_suite_and_keeps_remaining_conditions_unrun(self):
        self.check_invalid_suite('error')

    def check_invalid_suite(self, status):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'seeds':[1,2,3,4,5], 'conditions':[
                {'policy':'steve1','name':'wood','target_items':['oak_log']},
                {'policy':'steve1','name':'dirt','target_items':['dirt']}]}))
            argv=['probe','suite','--policy','steve1','--root',str(root),'--output',str(root/'run')]
            with patch('sys.argv',argv), patch('minecraftrl.cli.episode',return_value={'status':status,'execution_valid':False}) as run:
                self.assertEqual(main(),2)
            self.assertEqual(run.call_count,1)
            report=json.loads((root/'run/aggregate.json').read_text())
            self.assertIn('dirt',report)
            self.assertEqual(report['dirt']['unrun'],5)
            self.assertEqual(report['dirt']['attempted'],0)

if __name__ == '__main__': unittest.main()
