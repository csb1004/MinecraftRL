import unittest

from minecraftrl.pipeline.feedback import EPS_P, WINDOW_TICKS, Attempt, ProgressWindow, progress
from minecraftrl.pipeline.steps import Step

IRON = Step('mine-iron_ore', 'mine_iron_text', 'policy', 'iron_ore', 1, block='iron_ore')
CLIMB = Step('climb-2', 'climb_out_text', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})


def window(n, position=lambda i: (0.5, 64., 0.5), inventory=lambda i: {}, start=1000):
    return [(start + i, position(i), inventory(i)) for i in range(n)]


class ProgressTests(unittest.TestCase):
    def test_standing_still_without_items_has_no_progress(self):
        self.assertEqual(progress(window(600), IRON)['p'], 0.)

    def test_each_term_counts_a_third(self):
        # MineEvolve eq. 2 with equal weights: movement, inventory change and goal progress.
        moved = progress(window(600, position=lambda i: (0.5 + i / 20, 64., 0.5)), IRON)
        self.assertGreater(moved['move'], .9); self.assertAlmostEqual(moved['p'], moved['move'] / 3)
        picked = progress(window(600, inventory=lambda i: {'dirt': 3} if i > 300 else {}), IRON)
        self.assertAlmostEqual(picked['inventory'], .75); self.assertAlmostEqual(picked['p'], .25)
        mined = progress(window(600, inventory=lambda i: {'iron_ore': 1} if i > 300 else {}), IRON)
        self.assertAlmostEqual(mined['p'], (.5 + 1.) / 3)

    def test_a_jump_in_place_is_not_a_rise_but_a_pillar_is(self):
        # Check E3 (2026-11-24): forward+jump in the pit for 4,026 rows without gaining a level.
        jumping = window(600, position=lambda i: (0.5, 64. + (1.2 if i % 10 < 5 else 0.), 0.5))
        self.assertEqual(progress(jumping, CLIMB)['goal'], 0.)
        pillar = window(600, position=lambda i: (0.5, 64. + i // 100, 0.5))
        self.assertEqual(progress(pillar, CLIMB)['goal'], 1.)


class ProgressWindowTests(unittest.TestCase):
    def test_stagnation_fires_once_after_a_full_window(self):
        w = ProgressWindow(CLIMB)
        fired = [w.update(t, (0.5, 64. + (1.2 if t % 10 < 5 else 0.), 0.5), {'oak_planks': 3}) for t in range(1000, 1000 + 2 * WINDOW_TICKS)]
        hits = [(i, f) for i, f in enumerate(fired) if f]
        self.assertEqual(len(hits), 1)
        self.assertGreaterEqual(hits[0][0], WINDOW_TICKS); self.assertLess(hits[0][0], WINDOW_TICKS + 20)
        self.assertLess(hits[0][1]['p'], EPS_P); self.assertEqual(hits[0][1]['window_ticks'], WINDOW_TICKS)

    def test_wandering_or_gaining_is_not_stagnation(self):
        w = ProgressWindow(IRON)
        self.assertFalse(any(w.update(t, (0.5 + (t % 50) / 5, 64., 0.5 + (t // 50)), {}) for t in range(1000, 2500)))
        g = ProgressWindow(IRON)
        self.assertFalse(any(g.update(t, (0.5, 64., 0.5), {'iron_ore': 1} if t > 1300 else {}) for t in range(1000, 1700)))


class AttemptTests(unittest.TestCase):
    def test_feedback_has_the_typed_fields(self):
        # MineEvolve eq. 1: e = (z, y, ds, dv, f, p, l) per subgoal execution.
        a = Attempt(CLIMB, 1000, (0.5, 64., 0.5), {'oak_planks': 3})
        for t in range(1001, 1800): a.add(t, (0.5, 64. + (1.2 if t % 10 < 5 else 0.), 0.5), {'oak_planks': 3})
        e = a.finish('time_limit', 'time_limit')
        self.assertEqual((e['step_id'], e['skill'], e['success'], e['failure_type'], e['stagnant']),
                         ('climb-2', 'climb_out_text', False, 'time_limit', True))
        self.assertEqual(e['inventory_change'], {}); self.assertEqual(e['ticks'], 799)
        self.assertEqual(set(e['state_change']), {'start', 'end', 'height'})
        # Check F1 (2026-11-25): the attempt moved and placed dirt, then stood still for a window. The hold was for
        # stagnation, so the attempt is stagnant even though its whole-attempt progress is above the threshold.
        late = Attempt(CLIMB, 1000, (0.5, 64., 0.5), {'dirt': 9})
        for t in range(1001, 1040): late.add(t, (0.5 + (t - 1000) / 10, 64., 0.5), {'dirt': 9 - (t - 1000) // 10})
        for t in range(1040, 1700): late.add(t, (4.4, 64., 0.5), {'dirt': 6})
        e = late.finish('anomaly_hold', 'stagnation')
        self.assertGreater(e['progress'], .1); self.assertTrue(e['stagnant'])
        done = Attempt(IRON, 1000, (0.5, 64., 0.5), {})
        done.add(1010, (2.5, 64., 0.5), {'iron_ore': 1})
        e = done.finish('completed', None)
        self.assertEqual((e['success'], e['stagnant'], e['inventory_change']), (True, False, {'iron_ore': 1}))


if __name__ == '__main__':
    unittest.main()
