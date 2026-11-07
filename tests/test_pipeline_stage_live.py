import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.monitor_replay import load_run
from minecraftrl.pipeline.stage_replay import STAGES
from minecraftrl.pipeline.stage_live import LatentLog, LiveStageFeed
from test_pipeline_stage_replay import row, timeline, walk_rows, write_run

FIXTURE = {'rotate_with_facing': True, 'blocks': [{'dx': 0, 'dy': 1, 'dz': 8, 'block': 'iron_ore'}]}
START_T = .5   # timeline() starts the step at tick 10


class LiveStageFeedTest(unittest.TestCase):
    def decisions(self, rows):
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore'); views = []
        started = False
        for r in rows:
            if r['controller'] == 'policy':
                if not started: feed.start_step(START_T); started = True
                view = feed.decision_view(); views.append(None if view is None else view.stage)
            feed.observe(r)
        return feed, views

    def test_feed_matches_replay_stages_on_a_recorded_style_run(self):
        _, views = self.decisions([row(1, controller='prepare')] + walk_rows())
        replay = ['far', 'far', 'closing', 'near', 'aimed', 'mining', 'collect', 'far']   # stage_replay per policy row
        self.assertEqual(views[1:], replay[:-1])          # decision k saw row k-1

    def test_first_decision_uses_peek_of_the_prepare_row(self):
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore')
        feed.observe(row(1, controller='prepare')); feed.start_step(START_T)
        self.assertEqual(feed.decision_view().stage, 'far')
        self.assertEqual(list(feed.tracker.recent), [])

    def test_unreadable_row_is_an_unknown_stage_not_a_crash(self):
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore')
        feed.observe(row(1, controller='prepare')); feed.start_step(START_T)
        broken = row(2, controller='prepare'); broken['server_health'] = None; broken['yaw'] = None
        feed.observe(broken)
        self.assertIsNone(feed.decision_view())
        feed.observe(row(10)); self.assertEqual(feed.decision_view().stage, 'far')

    def test_no_view_before_the_fixture_is_applied(self):
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore')
        feed.observe(row(1, controller='prepare', fixture_state=1.)); feed.start_step(START_T)
        self.assertIsNone(feed.decision_view())

    def test_breaks_before_the_step_are_applied_without_a_collect(self):
        iron = (0, 65, 8)
        rows = [row(1, controller='prepare'),
                row(2, controller='prepare', destroys=1, destroyed='iron_ore', destroyed_pos=iron),
                row(30, pos=(0.5, 64., 5.5)), row(50, pos=(0.5, 64., 5.5))]
        _, views = self.decisions(rows)
        # Row 30 stands within reach of where the iron was: it is gone (far, not near) and broke before the step (no collect).
        self.assertEqual(views, ['far', 'far'])


class LatentLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.folder = Path(self.tmp.name) / 'r'

    def test_latent_log_flushes_partial_chunk_on_close(self):
        log = LatentLog(self.folder, width=4, cond_scale=6., meta={'arm': 'x'}, chunk=3)
        for k in range(1, 6): log.add(k, np.ones((2, 1, 4)) * k, 2 if k % 2 else None)
        manifest = log.close()
        self.assertEqual(manifest['rows'], 5); self.assertEqual(len(manifest['chunks']), 2); self.assertTrue(manifest['complete'])
        self.assertEqual(json.loads((self.folder / 'manifest.json').read_text()), manifest)
        second = np.load(self.folder / manifest['chunks'][1]['file'])
        self.assertEqual(second['step'].tolist(), [4, 5]); self.assertEqual(second['stage'].tolist(), [-1, 2])
        self.assertEqual(second['cond'].dtype, np.float16); self.assertEqual(second['cond'][1].tolist(), [5.] * 4)

    def test_manifest_is_current_after_each_chunk(self):
        log = LatentLog(self.folder, width=4, cond_scale=6., meta={}, chunk=2)
        for k in range(1, 4): log.add(k, np.zeros((2, 1, 4)), 0)
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        self.assertEqual((manifest['rows'], manifest['complete']), (2, False))

    def test_wrong_latent_shape_is_refused(self):
        log = LatentLog(self.folder, width=4, cond_scale=6., meta={})
        with self.assertRaises(ValueError): log.add(1, np.zeros((1, 1, 4)), 0)


if __name__ == '__main__':
    unittest.main()
