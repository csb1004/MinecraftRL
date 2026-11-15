"""Scoring building runs with the repair rule (design docs/specs/2026-11-15-build-repair-design.md §5)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.build_repair import repair_fixture
from minecraftrl.place_task import _at
from test_build_repair import ORIGIN, row, task

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location('score_build_repair', ROOT / 'scripts/score_build_repair.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.batch = Path(self.tmp.name)
        t = dict(task([(0, 0, 3)], strays=[(1, 0, 3)]), seed=1101)
        (self.batch / 'fixtures').mkdir()
        (self.batch / 'fixtures/task-1101.json').write_text(json.dumps(repair_fixture(t, teacher='careful')), encoding='utf-8')
        stray, cell = _at(ORIGIN, 0., 1, 0, 3), _at(ORIGIN, 0., 0, 0, 3)
        self.write('c4', 'c1101-s1-n1', [row(1), row(2, place=1, last=cell)], valid=True)
        self.write('c4', 'c1101-s1-n2', [row(1), row(2, place=1, last=cell),
                                         row(3, place=1, last=cell, destroys=1, destroyed=stray)], valid=True)
        self.write('c4', 'c1101-s1-n3', [row(1), row(2, place=1, last=cell),
                                         row(3, place=1, last=cell, destroys=1, destroyed=stray)], valid=False)

    def write(self, arm, name, rows, valid):
        run = self.batch / arm / name; run.mkdir(parents=True)
        (run / 'telemetry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        (run / 'summary.json').write_text(json.dumps(dict(execution_valid=valid, all_steps_completed=True)), encoding='utf-8')

    def test_a_game_succeeds_only_with_the_blueprint_filled_and_no_stray_left(self):
        s = load().score(self.batch, ['c4'])['c4']
        self.assertEqual((s['games'], s['valid'], s['success']), (3, 2, 1))
        self.assertEqual([(r['run'], r['success'], r['strays_left']) for r in s['runs']],
                         [('c1101-s1-n1', False, 1), ('c1101-s1-n2', True, 0), ('c1101-s1-n3', False, 0)])
        self.assertEqual((s['strays_left'], s['dug'], s['wrong'], s['broken']), (1, 1, 0, 0))   # valid games only

    def test_the_command_writes_the_scores(self):
        out = self.batch / 'repair-score.json'
        self.assertEqual(load().main(['--batch', str(self.batch), '--arms', 'c4', '--out', str(out)]), 0)
        self.assertEqual(json.loads(out.read_text())['c4']['success'], 1)


if __name__ == '__main__':
    unittest.main()
