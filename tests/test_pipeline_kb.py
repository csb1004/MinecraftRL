import sqlite3
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import Evidence, KnowledgeBase, condition_matches

WEB = Evidence('web', 'support', 'https://minecraft.wiki/w/Iron_Ore', 'requires a stone pickaxe')
OBS = Evidence('game_observation', 'support', 'results/run/telemetry.jsonl#1200')
OBS_REFUTE = Evidence('game_observation', 'refute', 'results/run/telemetry.jsonl#779')
CTX = {'game_version': '1.16.5'}
REQ = {'tool_class': 'pickaxe', 'min_tier': 'stone'}


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.path = Path(self.temp.name) / 'kb.sqlite'
        self.kb = KnowledgeBase(self.path)

    def tearDown(self):
        self.kb.close(); self.temp.cleanup()

    def test_new_claim_is_proposed_and_web_evidence_kept(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [WEB], run_id='r1')
        claim, = self.kb.query('iron_ore', 'harvest_requires', context=CTX)
        self.assertEqual((claim.id, claim.status, claim.object, claim.support, claim.refute), (cid, 'proposed', REQ, 1, 0))
        self.assertEqual(self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [OBS], run_id='r2'), cid)

    def test_game_observation_promotes_to_supported(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [WEB], run_id='r1')
        self.assertEqual(self.kb.add_evidence(cid, OBS, run_id='r1'), 'supported')

    def test_supported_can_be_disputed_later(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [OBS], run_id='r1')
        out = self.kb.resolve_conflict(cid, OBS_REFUTE, CTX, candidate_condition=None, observation_valid=True, run_id='r2')
        self.assertEqual(out, 'disputed')
        self.assertEqual(self.kb.query('iron_ore', 'harvest_requires', context=CTX), [])
        self.assertEqual(len(self.kb.query('iron_ore', 'harvest_requires', context=CTX, usable_only=False)), 1)

    def test_conflict_with_observable_condition_splits(self):
        cid = self.kb.add_claim('iron_ore', 'drops', 'raw_iron', {}, [WEB], run_id='r1')
        out = self.kb.resolve_conflict(cid, OBS_REFUTE, CTX, candidate_condition={'game_version': {'gte': '1.17'}},
                                       observation_valid=True, run_id='r1')
        self.assertEqual(out, 'conditionalized')
        self.assertEqual(self.kb.query('iron_ore', 'drops', context=CTX), [])
        self.assertEqual(len(self.kb.query('iron_ore', 'drops', context={'game_version': '1.18.2'})), 1)

    def test_unobservable_condition_key_not_used_for_split(self):
        cid = self.kb.add_claim('iron_ore', 'drops', 'raw_iron', {}, [WEB], run_id='r1')
        out = self.kb.resolve_conflict(cid, OBS_REFUTE, CTX, candidate_condition={'luck': 'high'}, observation_valid=True, run_id='r1')
        self.assertEqual(out, 'disputed')

    def test_invalid_observation_changes_nothing(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [OBS], run_id='r1')
        out = self.kb.resolve_conflict(cid, OBS_REFUTE, CTX, candidate_condition=None, observation_valid=False, run_id='r1')
        self.assertEqual(out, 'observation_error')
        self.assertEqual(self.kb.query('iron_ore', 'harvest_requires', context=CTX)[0].status, 'supported')

    def test_repeated_refutation_same_context_retracts(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [WEB], run_id='r1')
        for run in ('r2', 'r3'):
            out = self.kb.resolve_conflict(cid, OBS_REFUTE, CTX, candidate_condition=None, observation_valid=True, run_id=run)
        self.assertEqual(out, 'retracted')

    def test_history_preserved_after_retraction(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [WEB], run_id='r1')
        self.kb.set_status(cid, 'retracted', 'manual', run_id='r1')
        db = sqlite3.connect(self.path)
        self.assertEqual(db.execute('select to_status from claim_events where claim_id=?', (cid,)).fetchall()[-1], ('retracted',))
        self.assertEqual(db.execute('select count(*) from evidence where claim_id=?', (cid,)).fetchone(), (1,))
        db.close()

    def test_readonly_connection_rejects_writes(self):
        self.kb.add_claim('log', 'harvest_requires', {'tool_class': None, 'min_tier': 'hand'}, {}, [WEB], run_id='r1')
        with KnowledgeBase(self.path, readonly=True) as ro:
            self.assertEqual(len(ro.query('log', 'harvest_requires', context=CTX)), 1)
            with self.assertRaises(sqlite3.OperationalError):
                ro.add_claim('stone', 'harvest_requires', REQ, {}, [WEB], run_id='r1')

    def test_version_condition_compare(self):
        self.assertTrue(condition_matches({'game_version': {'lte': '1.16.5'}}, {'game_version': '1.16.5'}))
        self.assertFalse(condition_matches({'game_version': {'gte': '1.17'}}, {'game_version': '1.16.5'}))
        self.assertTrue(condition_matches({'game_version': {'gte': '1.17'}}, {'game_version': '1.18.2'}))
        self.assertFalse(condition_matches({'tool': 'stone_pickaxe'}, {'game_version': '1.16.5'}))

    def test_plan_dependents(self):
        cid = self.kb.add_claim('iron_ore', 'harvest_requires', REQ, {}, [WEB], run_id='r1')
        self.kb.record_plan('p1', [dict(id='iron', claim_ids=[cid]), dict(id='logs', claim_ids=[])])
        self.assertEqual(self.kb.dependents(cid), [('p1', 'iron')])

    def test_movement_relation_is_known(self):
        # Pit scenario knowledge (2026-11-04): a solid block in the path blocks movement.
        self.kb.add_claim('solid_block', 'blocks_movement', True, {}, [WEB], run_id='r1')
        self.assertEqual(len(self.kb.query('solid_block', 'blocks_movement', context=CTX)), 1)


if __name__ == '__main__':
    unittest.main()
