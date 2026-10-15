import subprocess,unittest
from types import SimpleNamespace
from unittest.mock import patch
from minecraftrl.preflight import command


class PreflightTimeoutTests(unittest.TestCase):
    def test_slow_local_java_can_have_a_bounded_startup_window(self):
        def slow(args,**kw):
            if kw['timeout']<23:raise subprocess.TimeoutExpired(args,kw['timeout'])
            return SimpleNamespace(returncode=0,stdout='',stderr='openjdk version "1.8.0_472"')
        with patch('minecraftrl.preflight.subprocess.run',side_effect=slow):
            self.assertIn('error',command(['java','-version']))
            self.assertEqual(command(['java','-version'],timeout_s=60)['exit_code'],0)


if __name__=='__main__':unittest.main()
