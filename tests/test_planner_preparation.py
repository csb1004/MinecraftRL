import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minecraftrl.world_store import digest


class PlannerPreparationTests(unittest.TestCase):
    def make(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name).resolve()
        source=root/'controller.py';source.write_text('prepared source',encoding='utf-8')
        folder=root/'results/planning-recovery-live-prepared-20261009';folder.mkdir(parents=True)
        manifest=folder/'manifest.json'
        manifest.write_text(json.dumps(dict(maximum_api_calls=1,api_retries=0,approved=False,
                                           files={'controller.py':digest(source)})),encoding='utf-8')
        spec=importlib.util.spec_from_file_location('prepared_planner_pilot',
            Path(__file__).resolve().parents[1]/'scripts/run_planner_exchange_pilot.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'run_rl_batch':SimpleNamespace(ROOT=root,rearm_allowed=lambda:None)}):
            spec.loader.exec_module(module)
        return root,source,manifest,module

    def test_preparation_is_read_only_and_changed_source_is_rejected(self):
        root,source,manifest,module=self.make()
        result=module.verify_prepared(root)
        self.assertEqual(result['sha256'],digest(manifest))
        self.assertFalse((root/'results/STOP').exists())
        source.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed'):module.verify_prepared(root)

    def test_invalid_call_budget_and_outside_path_are_rejected(self):
        root,source,manifest,module=self.make();original=json.loads(manifest.read_text())
        for update in ({'maximum_api_calls':2},{'api_retries':1},{'approved':True},
                       {'files':{'../outside.py':'a'*64}}):
            with self.subTest(update=update):
                manifest.write_text(json.dumps(dict(original,**update)),encoding='utf-8')
                with self.assertRaises(ValueError):module.verify_prepared(root)
