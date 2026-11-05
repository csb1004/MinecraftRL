"""Retired experiment commands must fail before any environment/API startup."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minecraftrl import cli


class OfficialRestartTests(unittest.TestCase):
    def test_legacy_commands_cannot_start_environment_or_load_policy(self):
        for extra in (['--rl-collect'], ['--rl-checkpoint', 'removed.pt'],
                      ['--rl-checkpoint=removed.pt'], ['--rl-world', 'removed.zip']):
            with self.subTest(extra=extra), patch('sys.argv', [
                'minecraftrl', 'episode', '--policy', 'steve1', *extra
            ]), patch.object(cli, 'episode') as episode, patch.object(cli, 'worker') as worker, \
                    patch.object(cli, 'inspect') as inspect:
                with self.assertRaisesRegex(SystemExit, 'Legacy RL experiment paths were removed'):
                    cli.main()
                episode.assert_not_called()
                worker.assert_not_called()
                inspect.assert_not_called()

    def test_official_preflight_still_routes_without_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('sys.argv', ['minecraftrl', 'preflight', '--policy', 'steve1',
                                   '--output', str(Path(folder)/'preflight')]), \
                patch.object(cli, 'inspect', return_value={'problems': []}) as inspect, \
                patch.object(cli, 'write_json'), patch.object(cli, 'episode') as episode, \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main()
            inspect.assert_called_once()
            episode.assert_not_called()
