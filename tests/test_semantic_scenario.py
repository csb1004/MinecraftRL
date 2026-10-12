import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class SemanticScenarioTests(unittest.TestCase):
    def test_fixture_uses_adapter_memory_and_confirmation_without_network(self):
        spec=importlib.util.spec_from_file_location('semantic_fixture',Path(__file__).resolve().parents[1]/'scripts/semantic_fixture.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'results'/'group'/'0'; folder.mkdir(parents=True)
            (folder/'events.jsonl').write_text('{"event":"session_ready"}\n')
            fixture=module.DialogueFixture(root,folder,folder.parent/'memory.json',restore=False)
            fixture.prepare()
            self.assertEqual(list(folder.glob('commands/*.json')),[])
            result=fixture.confirm()
            self.assertEqual(result['status'],'submitted')
            command=json.loads(next(folder.glob('commands/*.json')).read_text())
            self.assertEqual((command['op'],command['goal'],command['count']),('request','dirt',64))
            self.assertTrue(fixture.evidence['stale_confirmation_rejected'])
            self.assertTrue(fixture.evidence['summary_forwarded'])
            self.assertEqual(fixture.evidence['additional_paid_api_calls'],0)
