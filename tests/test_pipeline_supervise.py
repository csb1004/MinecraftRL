import json
import tempfile
import unittest
from pathlib import Path


class WorkerReportTests(unittest.TestCase):
    """The supervisor's reading of the worker's summary.json (2026-11-24: a startup timeout killed the worker while it
    was writing the summary; the empty file stopped the supervisor and left an empty summary behind)."""

    def test_an_empty_summary_after_a_startup_timeout_is_an_invalid_run(self):
        from minecraftrl.pipeline.cli import worker_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'; path.write_text('')
            report = worker_report(path, 'startup_timeout', -9)
        self.assertEqual(report['status'], 'startup_timeout')
        self.assertFalse(report['execution_valid'])
        self.assertTrue(report['shutdown_summary_unreadable'])

    def test_a_cut_summary_without_a_forced_reason_is_a_failed_worker(self):
        from minecraftrl.pipeline.cli import worker_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'; path.write_text('{"status": "comp')
            report = worker_report(path, None, 1)
        self.assertEqual(report['status'], 'worker_failed')
        self.assertFalse(report['execution_valid'])
        self.assertEqual(report['error'], 'worker exit 1')

    def test_a_missing_summary_keeps_the_missing_flag(self):
        from minecraftrl.pipeline.cli import worker_report
        with tempfile.TemporaryDirectory() as tmp:
            report = worker_report(Path(tmp) / 'summary.json', None, 2)
        self.assertEqual(report['status'], 'worker_failed')
        self.assertTrue(report['shutdown_summary_missing'])

    def test_a_written_summary_is_read_as_is(self):
        from minecraftrl.pipeline.cli import worker_report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            path.write_text(json.dumps(dict(status='completed', execution_valid=True, steps=3)))
            report = worker_report(path, None, 0)
        self.assertEqual(report, dict(status='completed', execution_valid=True, steps=3))


if __name__ == '__main__':
    unittest.main()
