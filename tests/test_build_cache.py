import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

module_path = Path(__file__).resolve().parents[1]/'scripts/minerl_build_cache.py'
spec = importlib.util.spec_from_file_location('minerl_build_cache', module_path)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class BuildCacheTests(unittest.TestCase):
    def test_complete_build_reused_but_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root/'minerl/MCP-Reborn/build/libs/mcprec-6.13.jar'
            calls = []
            def build_fixture():
                calls.append(1)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(artifact, 'w') as jar:
                    jar.writestr('net/minecraft/client/main/Main.class', b'test fixture')
            namespace = {'__file__':str(root/'setup.py'), 'pathlib':__import__('pathlib'),
                         'json':json, 'prep_mcp':build_fixture}
            code = cache.patch_setup('if True:\n    prep_mcp()\n')
            exec(code, namespace)
            exec(code, namespace)
            self.assertEqual(len(calls), 1)
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
                exec(code, namespace)
            self.assertEqual(len(calls), 1)

    def test_failed_build_never_creates_reuse_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fail():
                raise RuntimeError('compiler failed')
            namespace = {'__file__':str(root/'setup.py'), 'pathlib':__import__('pathlib'),
                         'json':json, 'prep_mcp':fail}
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                exec(cache.patch_setup('if True:\n    prep_mcp()\n'), namespace)
            self.assertFalse((root/'.minecraftrl-mcp-built.json').exists())
