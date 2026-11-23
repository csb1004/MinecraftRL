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

    def test_features_are_the_mining_features_of_the_decision_view(self):
        # Mining direction input design 2026-11-14 §1.
        from minecraftrl.reacquire_teacher import mining_features
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore')
        self.assertEqual(feed.features(), [0.] * 4)               # nothing seen yet
        feed.observe(row(1, controller='prepare')); feed.start_step(START_T)
        view = feed.decision_view()
        self.assertEqual(feed.features(), mining_features(feed.last_sample, view, 'iron_ore'))
        self.assertGreater(feed.features()[1], .9)                 # the iron is ahead of the start
        for r in walk_rows()[:3]: feed.observe(r)
        self.assertEqual(feed.features(), mining_features(feed.last_sample, feed.decision_view(), 'iron_ore'))

    def test_aim_feature_list_uses_the_aim_features_on_the_tracker_geometry(self):
        # Mining aim point design 2026-11-14 §1: a head trained on AIM_FEATURES gets them in the game too.
        from minecraftrl.reacquire_teacher import AIM_FEATURES, MINING_FEATURES, aim_features
        feed = LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore', feature_names=AIM_FEATURES)
        self.assertEqual(feed.features(), [0.] * 4)
        feed.observe(row(1, controller='prepare')); feed.start_step(START_T)
        for r in walk_rows()[:4]: feed.observe(r)
        view = feed.decision_view()
        self.assertEqual(feed.features(), aim_features(feed.last_sample, view, 'iron_ore', feed.tracker.geometry))
        self.assertEqual(LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore').feature_names, MINING_FEATURES)
        with self.assertRaises(ValueError): LiveStageFeed(FIXTURE, 'iron_ore', 'iron_ore', feature_names=('a',))

    def test_hook_hands_the_head_feature_names_to_its_own_feed(self):
        from types import SimpleNamespace
        from minecraftrl.pipeline.stage_live import ReacquireHook
        from minecraftrl.reacquire_teacher import AIM_FEATURES, MINING_FEATURES
        for names in (AIM_FEATURES, MINING_FEATURES):
            with self.subTest(names=names):
                head = SimpleNamespace(width=4, features=names, stage=None, feature=None, active=False)
                hook = ReacquireHook(head, FIXTURE, 'iron_ore', 'iron_ore', None, cond_scale=6., meta={})
                self.assertEqual(hook.feed.feature_names, names)
        plain = ReacquireHook(SimpleNamespace(width=4), FIXTURE, 'iron_ore', 'iron_ore', None, cond_scale=6., meta={})
        self.assertEqual(plain.feed.feature_names, MINING_FEATURES)

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

    def test_feature_values_are_recorded_per_decision(self):
        # Jump direction input design 2026-11-14 §2: the features the head saw are kept next to the stage.
        log = LatentLog(self.folder, width=4, cond_scale=6., meta={}, features=('a', 'b'))
        log.add(1, np.zeros((2, 1, 4)), 0, features=[.5, -1.]); log.add(2, np.zeros((2, 1, 4)), None, features=None)
        manifest = log.close()
        self.assertEqual(manifest['features'], ['a', 'b'])
        with np.load(self.folder / manifest['chunks'][0]['file']) as data:
            self.assertEqual(data['features'].dtype, np.float32)
            self.assertEqual(data['features'].tolist(), [[.5, -1.], [0., 0.]])
        with self.assertRaises(ValueError): log.add(3, np.zeros((2, 1, 4)), 0, features=[1.])
        plain = LatentLog(Path(self.tmp.name) / 'p', width=4, cond_scale=6., meta={})
        plain.add(1, np.zeros((2, 1, 4)), 0); manifest = plain.close()
        self.assertNotIn('features', manifest)                    # records without features keep the earlier format
        with np.load(Path(self.tmp.name) / 'p' / manifest['chunks'][0]['file']) as data: self.assertNotIn('features', data)

    def test_hook_gives_the_head_the_feed_features_and_logs_them(self):
        from types import SimpleNamespace
        from minecraftrl.pipeline.stage_live import ReacquireHook

        class Feed:
            def observe(self, row): pass
            def start_step(self, t): pass
            def decision_view(self): return SimpleNamespace(stage='run')
            def features(self): return [.25, .75]

        head = SimpleNamespace(stages=('turn', 'run'), width=4, features=('a', 'b'), stage=None, feature=None,
                               active=False, last_latent=np.zeros((2, 1, 4)))
        hook = ReacquireHook(head, {}, None, None, self.folder, cond_scale=6., meta={}, feed=Feed())
        hook.before_act()
        self.assertEqual((head.stage, head.feature), (1, [.25, .75]))
        hook.after_dispatch(7); manifest = hook.log.close()
        with np.load(self.folder / manifest['chunks'][0]['file']) as data:
            self.assertEqual(data['features'].tolist(), [[.25, .75]])
        hook.activate(False)
        self.assertEqual((head.stage, head.feature), (None, None))

    def test_closing_reports_the_rows_an_attack_decode_changed(self):
        # Comparison arm of 2026-11-24: the summary says how many rows the decode pressed or released attack on.
        from types import SimpleNamespace
        from minecraftrl.pipeline.stage_live import ReacquireHook

        class Feed:
            def observe(self, row): pass
            def start_step(self, t): pass
            def decision_view(self): return None
        stats = dict(rule='r', rows=3, pressed=2, released=0, stages=['dig'])
        head = SimpleNamespace(stages=('dig',), width=4, features=(), stage=None, feature=None, active=False,
                               decode_stats=lambda: stats)
        self.assertEqual(ReacquireHook(head, {}, None, None, None, cond_scale=6., meta={}, feed=Feed()).close(),
                         dict(decisions=0, latent_rows=0, attack_decode_rows=stats))
        head.decode_stats = lambda: None
        self.assertEqual(ReacquireHook(head, {}, None, None, None, cond_scale=6., meta={}, feed=Feed()).close(),
                         dict(decisions=0, latent_rows=0))


if __name__ == '__main__':
    unittest.main()
