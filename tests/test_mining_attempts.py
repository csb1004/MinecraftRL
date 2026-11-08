import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.mining_attempts import attempts, destroyed_share, dev_verdict, pool, r0_gate, run_block


def r(hit='air', mining='air', attack=1, destroys=0, destroyed='air', controller='policy'):
    return dict(controller=controller, action=dict(attack=attack),
                demo=dict(demo_hit_block=hit, demo_mining_block=mining, demo_destroys=float(destroys),
                          demo_last_destroyed_block=destroyed))


def arm(success, median_s, destroyed, other):
    return dict(success=success, median_s=median_s,
                attempts=dict(on_rows=0, attack_rows=0, ends=dict(destroyed=destroyed, released=other, left=0),
                              lengths=[], streaks=[]))


class AttemptsTest(unittest.TestCase):
    def test_destroyed_released_and_left_are_told_apart(self):
        rows = [r(), r('coal_ore', 'coal_ore'), r('coal_ore', 'coal_ore'), r('coal_ore', 'coal_ore', destroys=1, destroyed='coal_ore'),
                r('coal_ore', 'coal_ore', destroys=1), r('coal_ore', 'air', attack=0, destroys=1),      # released on the block
                r('coal_ore', 'coal_ore', destroys=1), r('stone', 'air', destroys=1)]                    # crosshair left
        s = attempts(rows, 'coal_ore')
        self.assertEqual(s['ends'], dict(destroyed=1, released=1, left=1))
        self.assertEqual(s['lengths'], [2, 1, 1]); self.assertEqual(s['streaks'], [3])
        self.assertEqual((s['on_rows'], s['attack_rows']), (6, 5))
        self.assertAlmostEqual(destroyed_share(s), 1 / 3)

    def test_other_blocks_and_non_policy_rows_do_not_count(self):
        rows = [r('stone', 'stone'), r('stone', 'stone', destroys=1, destroyed='stone'),
                r('coal_ore', 'coal_ore', controller='prepare'), r('coal_ore', 'air', controller='prepare')]
        s = attempts(rows, 'coal_ore')
        self.assertEqual(s['ends'], dict(destroyed=0, released=0, left=0)); self.assertEqual(s['on_rows'], 0)
        self.assertEqual(destroyed_share(s), 0.)

    def test_log_family_matches_any_log(self):
        rows = [r('birch_log', 'birch_log'), r('birch_log', 'birch_log', destroys=1, destroyed='birch_log')]
        self.assertEqual(attempts(rows, 'log')['ends']['destroyed'], 1)

    def test_pool_adds_runs(self):
        a = attempts([r('sand', 'sand'), r('sand', 'sand', destroys=1, destroyed='sand')], 'sand')
        p = pool([a, a])
        self.assertEqual(p['ends']['destroyed'], 2); self.assertEqual(p['on_rows'], 4)


class GateAndVerdictTest(unittest.TestCase):
    def test_gate_needs_two_blocks_below_half(self):
        def s(d, o): return dict(on_rows=0, attack_rows=0, ends=dict(destroyed=d, released=o, left=0), lengths=[], streaks=[])
        self.assertFalse(r0_gate(dict(a=s(1, 1), b=s(1, 1), c=s(0, 3)))['passed'])     # exactly half is not failing
        g = r0_gate(dict(a=s(1, 2), b=s(0, 0), c=s(3, 0)))
        self.assertTrue(g['passed']); self.assertEqual(g['failing_blocks'], ['a', 'b'])

    def test_verdict_needs_all_three_conditions(self):
        official, iron_only = arm(2, 120., 1, 9), arm(3, 100., 2, 8)
        good = dev_verdict(official, arm(2, 110., 3, 7), iron_only)
        self.assertTrue(good['improved'])
        self.assertFalse(dev_verdict(official, arm(1, 110., 3, 7), iron_only)['conditions']['success_not_lower'])
        self.assertFalse(dev_verdict(official, arm(2, 110., 1, 4), iron_only)['conditions']['destroyed_share'])  # 0.2 < 2 x 0.1? equal -> not above iron-only
        self.assertFalse(dev_verdict(official, arm(2, 130., 3, 7), iron_only)['conditions']['median_not_longer'])
        self.assertFalse(dev_verdict(official, arm(2, None, 3, 7), iron_only)['conditions']['median_not_longer'])
        self.assertTrue(dev_verdict(arm(0, None, 0, 5), arm(1, 150., 1, 1))['improved'])   # final form, no iron-only arm

    def test_zero_official_share_still_needs_a_higher_share(self):
        self.assertFalse(dev_verdict(arm(0, None, 0, 5), arm(0, None, 0, 5))['conditions']['destroyed_share'])


class RunBlockTest(unittest.TestCase):
    def test_block_comes_from_the_recorded_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'summary.json').write_text(json.dumps(dict(execution_valid=True, all_steps_completed=True)))
            (run / 'request.json').write_text(json.dumps(dict(plan=[dict(id='sand', target='sand', block='sand', count=1)])))
            (run / 'telemetry.jsonl').write_text(json.dumps(r('sand', 'sand')) + '\n'
                                                 + json.dumps(r('sand', 'sand', destroys=1, destroyed='sand')) + '\n')
            block, success, s = run_block(run)
            self.assertEqual((block, success, s['ends']['destroyed']), ('sand', True, 1))


if __name__ == '__main__':
    unittest.main()
