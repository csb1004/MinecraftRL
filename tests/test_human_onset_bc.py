"""Raw-frame transition labels and fixed-mass, train-only BC weighting."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from minecraftrl.human_onset_bc import jump_transitions, load_train_transitions, reweight_onsets
from minecraftrl.world_store import digest


def frame(tick, jump=False):
    return dict(tick=tick, isGuiOpen=False,
                keyboard=dict(keys=['key.keyboard.space'] if jump else []))


class HumanOnsetBCTests(unittest.TestCase):
    def test_uses_preceding_raw_frame_not_previous_retained_label(self):
        raw = [frame(0), frame(1, True), frame(2, True), frame(3), frame(4, True)]
        self.assertEqual(jump_transitions(raw, [2, 4]), ['continuation', 'onset'])
        self.assertEqual(jump_transitions(raw, [1, 2, 3]), ['onset', 'continuation', 'nonjump'])

    def test_requires_unique_indices_and_real_consecutive_history(self):
        raw = [frame(0), frame(1, True)]
        for indices in ([0], [-1], [2], [True], [1, 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                jump_transitions(raw, indices)
        raw[1]['tick'] = 3
        with self.assertRaises(ValueError):
            jump_transitions(raw, [1])

    def test_gui_or_invalid_keys_rejected(self):
        for index in (0, 1):
            raw = [frame(0), frame(1, True)]
            raw[index]['isGuiOpen'] = True
            with self.assertRaises(ValueError):
                jump_transitions(raw, [1])
        raw = [frame(0), frame(1, True)]
        raw[1]['keyboard']['keys'] = 'key.keyboard.space'
        with self.assertRaises(ValueError):
            jump_transitions(raw, [1])

    def test_default_is_exact_identity_even_without_onsets(self):
        weights = [0.123456789, 4.5, 0.04]
        result, receipt = reweight_onsets(weights, [False, True, True], ['nonjump', 'continuation'], 1)
        self.assertEqual(result, weights)
        self.assertTrue(receipt['default_identity'])

    def test_unequal_bucket_weights_keep_local_values_and_human_mass(self):
        weights = [1.3, 0.2, 2.7, 3., 0.8]
        flags = [False, True, False, True, True]
        result, receipt = reweight_onsets(weights, flags, ['onset', 'continuation', 'nonjump'], 4)
        self.assertEqual([result[0], result[2]], [weights[0], weights[2]])
        self.assertAlmostEqual(math.fsum(result[i] for i in (1, 3, 4)), 4.)
        self.assertAlmostEqual((result[1]/result[3])/(weights[1]/weights[3]), 4.)
        self.assertAlmostEqual(result[3]/result[4], weights[3]/weights[4])
        self.assertTrue(receipt['local_weights_unchanged'])
        self.assertNotEqual(result, weights)

    def test_invalid_weighting_rejected(self):
        for factor in (0, .5, 9, True, float('nan'), float('inf')):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                reweight_onsets([1., 1.], [False, True], ['onset'], factor)
        for weights, flags, kinds in (
                ([1., 0.], [False, True], ['onset']),
                ([1., float('nan')], [False, True], ['onset']),
                ([1., float('inf')], [False, True], ['onset']),
                ([1., 1.], [False], ['onset']),
                ([1., 1.], [False, 1], ['onset']),
                ([1., 1.], [True, True], ['onset', 'nonjump']),
                ([1., 1.], [False, True], []),
                ([1., 1.], [False, True], ['nonjump'])):
            with self.subTest(weights=weights, flags=flags, kinds=kinds), self.assertRaises(ValueError):
                reweight_onsets(weights, flags, kinds, 4)

    def test_train_provenance_and_development_rejected_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'raw.jsonl'
            source.write_text('\n'.join(json.dumps(r) for r in [frame(0), frame(1, True), frame(2, True)]), encoding='utf-8')
            pair = dict(session='train-a', split='train', jsonl='raw.jsonl', jsonl_sha256=digest(source))
            manifest = root/'pairs.json'
            manifest.write_text(json.dumps(dict(pairs=[pair])), encoding='utf-8')
            contract = dict(pair_manifest=dict(path='pairs.json', sha256=digest(manifest)))
            rows = [dict(owner='human', bc_source='hindsight_reference', human_session='train-a', human_frame=i) for i in (1, 2)]
            kinds, receipt = load_train_transitions(root, contract, rows)
            self.assertEqual(kinds, ['onset', 'continuation'])
            self.assertEqual(receipt['counts'], dict(onset=1, continuation=1, nonjump=0))
            self.assertFalse(receipt['development_used'])
            pair.update(split='development', jsonl='missing-dev-source.jsonl')
            manifest.write_text(json.dumps(dict(pairs=[pair])), encoding='utf-8')
            contract['pair_manifest']['sha256'] = digest(manifest)
            with self.assertRaisesRegex(ValueError, 'cannot inspect development'):
                load_train_transitions(root, contract, rows)


if __name__ == '__main__':
    unittest.main()
