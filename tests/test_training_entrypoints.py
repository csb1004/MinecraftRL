"""Adapter CLI safety and source provenance, without a game or GPU updates."""
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class TrainingEntrypointTests(unittest.TestCase):
    def test_existing_recovery_import_remains_compatible(self):
        from minecraftrl.training import adapter, validation
        scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
        with patch.object(sys, 'path', [scripts, *sys.path]):
            legacy = importlib.import_module('train_recovery_adapter')
        self.assertIs(legacy.main, adapter.main)
        self.assertIs(legacy.validate, validation.validate)

    def test_all_profiles_stop_before_loading_data_or_creating_output(self):
        from minecraftrl import effect_learning, execution_learning
        from minecraftrl.training import adapter
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / 'STOP'
            stop.write_text('keep stopped', encoding='utf-8')
            for profile in (None, effect_learning, execution_learning):
                output = root / 'results/candidate'
                argv = ['--root', str(root), '--output', str(output), '--stop-file', str(stop),
                        '--runs', str(root / 'results/missing'), '--variant', 'nonlinear-on']
                with self.subTest(profile=profile), patch.object(adapter.torch, 'load') as load:
                    with self.assertRaises(InterruptedError):
                        adapter.main(profile, argv=argv)
                    load.assert_not_called()
                    self.assertFalse(output.exists())
            self.assertEqual(stop.read_text(), 'keep stopped')

    def test_e2_only_options_and_no_implicit_training(self):
        from minecraftrl import execution_learning
        from minecraftrl.training.adapter import build_parser
        common = ['--root', '.', '--output', 'results/new', '--stop-file', 'results/STOP',
                  '--runs', 'results/train', '--variant', 'nonlinear-on']
        args = build_parser(execution_learning).parse_args(common)
        self.assertEqual(args.critic_warmup_epochs, 0)
        self.assertEqual(args.retention_runs, [])
        self.assertIsNone(args.parent)
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--critic-warmup-epochs', '11'], ['--variant', 'nonlinear-off']):
                with self.assertRaises(SystemExit):
                    build_parser(execution_learning).parse_args(common + extra)
            with self.assertRaises(SystemExit):
                build_parser().parse_args(common + ['--critic-warmup-epochs', '1'])


class TrainingProvenanceTests(unittest.TestCase):
    def test_nested_implementation_and_entrypoints_are_hashed_without_artifacts(self):
        from minecraftrl.training.provenance import source_manifest, verify_sources
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {'src/minecraftrl/training/adapter.py': b'implementation',
                        'src/minecraftrl/rl_opt.py': b'loss',
                        'scripts/train_execution_adapter.py': b'entrypoint',
                        'pyproject.toml': b'[project]',
                        'results/secret.json': b'private',
                        'weights/model.pt': b'large',
                        'src/minecraftrl/__pycache__/x.pyc': b'cache'}
            for name, raw in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            before = source_manifest(root)
            verify_sources(root, before)
            files = {row['path']: row['sha256'] for row in before['files']}
            self.assertEqual(set(files), set(list(contents)[:4]))
            for name, digest in files.items():
                self.assertEqual(digest, hashlib.sha256(contents[name]).hexdigest())
            implementation = root / 'src/minecraftrl/training/adapter.py'
            implementation.write_bytes(b'changed')
            after = source_manifest(root)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_sources(root, before)
            self.assertNotEqual(before, after)
            self.assertEqual(after, source_manifest(root))
            self.assertEqual([r['path'] for r in after['files']], sorted(files))


if __name__ == '__main__':
    unittest.main()
