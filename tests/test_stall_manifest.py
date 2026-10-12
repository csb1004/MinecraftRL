import copy
import importlib.util
import json
from pathlib import Path
import unittest


root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stall_replay_cli',root/'scripts/replay_stall_detection.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class StallManifestTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((root/'configs/graph-stall-replay-20261008.json').read_text())

    def test_frozen_manifest_valid(self):
        cli.validate_manifest(self.config)

    def test_output_traversal_rejected(self):
        self.config['runs'][0]['id']='../outside'
        with self.assertRaises(ValueError):cli.validate_manifest(self.config)

    def test_missing_hash_rejected(self):
        del self.config['runs'][0]['sha256']['events.jsonl']
        with self.assertRaises(ValueError):cli.validate_manifest(self.config)

    def test_final_seed_cannot_be_removed_from_exclusions(self):
        self.config['excluded_final_seeds']=[]
        with self.assertRaises(ValueError):cli.validate_manifest(self.config)

    def test_duplicate_id_rejected(self):
        self.config['runs'].append(copy.deepcopy(self.config['runs'][0]))
        with self.assertRaises(ValueError):cli.validate_manifest(self.config)


if __name__=='__main__':unittest.main()
