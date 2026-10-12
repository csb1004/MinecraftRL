import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from minecraftrl.rl_control import claim_automatic_stop,publish_completion


class RLControlTests(unittest.TestCase):
    def test_completion_cannot_overwrite_a_late_manual_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            stop=Path(directory)/'STOP'
            stop.write_text('late manual stop')
            publish_completion(stop,'RL collection ended')
            self.assertEqual(stop.read_text(),'late manual stop')
            fresh=Path(directory)/'fresh'
            publish_completion(fresh,'RL training ended')
            self.assertEqual(fresh.read_text(),'RL training ended')

    def test_only_claims_automatic_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            stop=Path(directory)/'STOP';claimed=Path(directory)/'previous'
            stop.write_text('user requested stop')
            with self.assertRaises(InterruptedError):claim_automatic_stop(stop,claimed)
            self.assertEqual(stop.read_text(),'user requested stop')
            stop.write_text('RL collection ended; controlled re-arm required.')
            claimed=Path(directory)/'next-previous'
            claim_automatic_stop(stop,claimed)
            self.assertFalse(stop.exists());self.assertTrue(claimed.exists())

    def test_new_stop_after_atomic_claim_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            stop=Path(directory)/'STOP';claimed=Path(directory)/'previous'
            stop.write_text('RL collection ended; controlled re-arm required.')
            rename=Path.rename
            def concurrent_stop(path,destination):
                result=rename(path,destination);stop.write_text('new manual stop');return result
            with patch.object(Path,'rename',concurrent_stop):
                with self.assertRaises(InterruptedError):claim_automatic_stop(stop,claimed)
            self.assertEqual(stop.read_text(),'new manual stop')
