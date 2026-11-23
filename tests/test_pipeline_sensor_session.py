"""The opt-in sensor path inside DemoSession (sensor-tier design 2026-11-19 §2.5, §3.1; plan 2026-11-22 tasks 2 and 4)."""
import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.kb import Evidence, KnowledgeBase
from minecraftrl.pipeline.oversight import Oversight
from minecraftrl.pipeline.recorder import Recorder
from minecraftrl.pipeline.sensor import SensorLog
from minecraftrl.pipeline.session import DemoSession
from minecraftrl.pipeline.steps import Step
from minecraftrl.runtime import StopGate
from test_pipeline_sensor import raw_sensor
from test_pipeline_session import SKILLS, Clock, FakeController, FakeEnv, FakePolicy, info, noop

IRON = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 3, block='iron_ore')]


def sensed(n, *, selected='air', **sensor):
    value = info(tick=1000 + n, selected=selected, hit='stone')
    value['workshop'].update(client_x=0.5, client_y=64., client_z=0.5, client_on_ground=1, server_health=20., yaw=0.)
    value['sensor'] = raw_sensor(sensor_client_tick=1000 + n, **sensor)
    return value


class SensorSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.out = Path(self.temp.name); self.gate = StopGate()
        self.kb = KnowledgeBase(self.out / 'kb.sqlite'); self.addCleanup(self.kb.close)
        self.kb.add_claim('iron_ore', 'harvest_requires', dict(tool_class='pickaxe', min_tier='stone'), {},
                          [Evidence('web', 'support', 'https://minecraft.wiki')], run_id='seed')

    def tearDown(self):
        self.recorder.close(); self.temp.cleanup()

    def run_session(self, env, action, *, sensor=True, clock=None):
        self.recorder = Recorder(self.out / 'run')
        oversight = Oversight(self.kb, {'game_version': '1.16.5'}, 'iron_ore', IRON, run_id='test',
                              selected={'iron_ore': 'dig'})
        log = SensorLog(self.out / 'run/sensor') if sensor else None
        session = DemoSession(env, FakePolicy(action), None, self.recorder, self.gate, self.out / 'STOP', limit_s=60.,
                              skills=SKILLS, clock=clock or Clock(), rule_controller=FakeController, settle_steps=0,
                              oversight=oversight, sensor_log=log)
        return session.run(IRON)

    def rows(self):
        return [json.loads(line) for line in (self.out / 'run/telemetry.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_dropping_a_needed_item_never_reaches_the_game_and_holds_when_it_persists(self):
        action = noop(); action['drop'] = 1
        env = FakeEnv(lambda n, a: sensed(n, selected='iron_ore'))
        report = self.run_session(env, action, clock=Clock(.1))
        self.assertEqual(sum(a['drop'] for a in env.actions), 0)
        result = report['stage_results'][0]
        self.assertEqual((report['status'], result['anomaly']['kind']), ('anomaly_hold', 'guard_drop_needed_persistent'))
        self.assertGreater(result['sensor_guard_masks'], 0)
        policy_rows = [r for r in self.rows() if r.get('controller') == 'policy']
        self.assertTrue(policy_rows and all(r['guard_masked'] == ['drop'] for r in policy_rows))
        self.assertTrue(all('sensor_row' in r for r in self.rows()))
        manifest = json.loads((self.out / 'run/sensor/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((manifest['schema'], manifest['complete'], manifest['rows']), ('sensor_v1', True, env.calls))
        self.assertEqual(report['sensor']['rows'], env.calls)

    def test_a_worn_tool_holds_before_the_attack(self):
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: sensed(n, selected='stone_pickaxe', sensor_tool_damage=129))   # 2 uses left, 3 needed
        report = self.run_session(env, action)
        result = report['stage_results'][0]
        self.assertEqual((report['status'], result['anomaly']['kind']), ('anomaly_hold', 'guard_tool_wear'))
        self.assertEqual(sum(a['attack'] for a in env.actions), 0)

    def test_the_sensor_build_feed_gets_every_frame_and_needs_the_sensor_log(self):
        from minecraftrl.build_sensor import SensorBuildFeed, sensor_fixture
        from minecraftrl.build_task import generate_build
        fixture = sensor_fixture(generate_build(103))
        self.recorder = Recorder(self.out / 'run')
        common = dict(limit_s=60., skills=SKILLS, clock=Clock(), rule_controller=FakeController, settle_steps=0, build=fixture)
        with self.assertRaises(ValueError):
            DemoSession(FakeEnv(lambda n, a: sensed(n)), FakePolicy(noop()), None, self.recorder, self.gate, self.out / 'STOP', **common)
        session = DemoSession(FakeEnv(lambda n, a: sensed(n)), FakePolicy(noop()), None, self.recorder, self.gate,
                              self.out / 'STOP', sensor_log=SensorLog(self.out / 'run/sensor'), **common)
        self.assertIsInstance(session.build, SensorBuildFeed)
        session.obs = session.env.reset()
        self.assertTrue(session.dispatch(noop(), 'prepare'))
        self.assertIsNotNone(session.build.last_frame)
        self.assertIsNotNone(session.build.decision_view())

    def test_without_the_sensor_log_nothing_changes(self):
        action = noop(); action['drop'] = 1
        env = FakeEnv(lambda n, a: sensed(n, selected='iron_ore'))
        report = self.run_session(env, action, sensor=False, clock=Clock(.1))
        self.assertGreater(sum(a['drop'] for a in env.actions), 0)          # the earlier path has no drop guard
        self.assertFalse(any('sensor_row' in r or 'guard_masked' in r for r in self.rows()))
        self.assertNotIn('sensor', report)
        self.assertNotIn('sensor_guard_masks', report['stage_results'][0])


class BlockSelectTests(unittest.TestCase):
    """Multi-material building (user request 2026-11-23, beacon demo): a disclosed rule selects the hotbar slot that
    holds the current cell's block before the head acts; a neutral hold after success records the result."""
    BUILD = [Step('build', 'dig', 'policy', 'build_done', 1, params=dict(until='build'))]

    def setUp(self):
        from minecraftrl.build_beacon import beacon_build
        from minecraftrl.build_sensor import sensor_fixture
        self.temp = tempfile.TemporaryDirectory(); self.out = Path(self.temp.name); self.gate = StopGate()
        self.fixture = sensor_fixture(beacon_build(1501)); self.recorder = Recorder(self.out / 'run')

    def tearDown(self):
        self.recorder.close(); self.temp.cleanup()

    def env(self, switches=True):
        """Sensor frames whose held item follows the hotbar keys the session sends (the fixture's gifts fill slots 1-4)."""
        gifts = [g['item'] for g in self.fixture['give']]; held = ['air']

        def script(n, action):
            for i, item in enumerate(gifts):
                if switches and action.get('hotbar.%d' % (i + 1)): held[0] = item
            return sensed(n, selected=held[0])
        return FakeEnv(script)

    def session(self, env, policy, fixture=None):
        return DemoSession(env, policy, None, self.recorder, self.gate, self.out / 'STOP', limit_s=6., skills=SKILLS,
                           clock=Clock(.05), rule_controller=FakeController, settle_steps=0, build=fixture or self.fixture,
                           sensor_log=SensorLog(self.out / 'run/sensor'))

    def rows(self):
        return [json.loads(line) for line in (self.out / 'run/telemetry.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_the_rule_selects_the_needed_block_before_the_head_acts(self):
        policy = FakePolicy(noop()); report = self.session(self.env(), policy).run(self.BUILD)
        rows = [r for r in self.rows() if r['controller'] in ('rule_select', 'policy')]
        self.assertEqual(rows[0]['controller'], 'rule_select')
        self.assertEqual(rows[0]['action'], {'camera': [0., 0.], 'hotbar.1': 1})        # gold sits in slot 1
        self.assertTrue(len(rows) > 2 and all(r['controller'] == 'policy' for r in rows[1:]))
        self.assertIn(policy.calls - (len(rows) - 1), (0, 1))   # the last act may meet the deadline before dispatch
        self.assertEqual(report['stage_results'][0]['block_selects'], 1)

    def test_a_selection_that_never_takes_ends_the_step(self):
        from minecraftrl.pipeline.session import SELECT_LIMIT
        policy = FakePolicy(noop()); report = self.session(self.env(switches=False), policy).run(self.BUILD)
        result = report['stage_results'][0]
        self.assertEqual((result['status'], result['block_selects'], policy.calls), ('block_select_failed', SELECT_LIMIT, 0))

    def test_single_kind_tasks_have_no_selection_rule(self):
        from minecraftrl.build_sensor import sensor_fixture
        from minecraftrl.build_task import generate_build
        policy = FakePolicy(noop())
        report = self.session(self.env(), policy, sensor_fixture(generate_build(103))).run(self.BUILD)
        self.assertFalse(any(r['controller'] == 'rule_select' for r in self.rows()))
        self.assertNotIn('block_selects', report['stage_results'][0])

    def test_the_hold_after_success_sends_only_neutral_input(self):
        session = self.session(self.env(), FakePolicy(noop()))
        session.obs = session.env.reset(); session.started = session.clock()
        step = Step('build', 'dig', 'policy', 'build_done', 1, params=dict(until='build', hold_after_steps=3))
        self.assertEqual(session.hold_after(step), 3)
        rows = [r for r in self.rows() if r['controller'] == 'after_success']
        self.assertEqual(len(rows), 3); self.assertTrue(all(r['action'] == {'camera': [0., 0.]} for r in rows))
        self.assertEqual(session.hold_after(Step('b', 'dig', 'policy', 'build_done', 1, params=dict(until='build'))), 0)
        self.gate.stop('stop_file')
        self.assertEqual(session.hold_after(step), 0)


class RepairSessionTests(unittest.TestCase):
    """Sensor repair (2026-11-23): the rule reads the client hand, which shows a pressed slot on the same row, finds a
    slot whose item moved (a picked-up block lands in the first empty slot), and the monitor's events reach the timeline."""
    BUILD = BlockSelectTests.BUILD

    def setUp(self):
        from minecraftrl.build_sensor_repair import SENSOR_DIG, dig_task, repair_fixture
        self.temp = tempfile.TemporaryDirectory(); self.out = Path(self.temp.name); self.gate = StopGate()
        self.fixture = repair_fixture(dig_task(1601), SENSOR_DIG); self.recorder = Recorder(self.out / 'run')

    def tearDown(self):
        self.recorder.close(); self.temp.cleanup()

    def env(self, hotbar):
        """Sensor frames whose client hand item is the pressed slot's item in `hotbar` (slots 1-9); the server's selected
        item never changes, as if it lagged."""
        held = ['air']

        def script(n, action):
            for i, item in enumerate(hotbar):
                if action.get('hotbar.%d' % (i + 1)): held[0] = item
            return sensed(n, selected='air', sensor_tool_item=held[0])
        return FakeEnv(script)

    def session(self, env, want):
        from minecraftrl.pipeline.session import DemoSession as Session
        s = Session(env, FakePolicy(noop()), None, self.recorder, self.gate, self.out / 'STOP', limit_s=2., skills=SKILLS,
                    clock=Clock(.05), rule_controller=FakeController, settle_steps=0, build=self.fixture,
                    sensor_log=SensorLog(self.out / 'run/sensor'))
        s.build.needed_kind = lambda: want
        return s

    def presses(self):
        rows = [json.loads(line) for line in (self.out / 'run/telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
        return [next(k for k in r['action'] if k.startswith('hotbar')) for r in rows if r['controller'] == 'rule_select']

    def test_one_press_puts_the_pickaxe_in_hand_from_the_client_hand(self):
        from minecraftrl.build_sensor_repair import DIG_KINDS, PICKAXE
        session = self.session(self.env(list(DIG_KINDS) + [PICKAXE]), PICKAXE)
        self.assertEqual(session.block_slots[PICKAXE], 'hotbar.5')
        report = session.run(self.BUILD)
        self.assertEqual(self.presses(), ['hotbar.5'])
        self.assertEqual(report['stage_results'][0]['block_selects'], 1)

    def test_a_moved_item_is_found_in_the_next_unknown_slot_and_the_slot_map_learns(self):
        from minecraftrl.build_sensor_repair import PICKAXE
        hotbar = ['air', 'iron_block', 'diamond_block', 'beacon', 'gold_block', PICKAXE]
        session = self.session(self.env(hotbar), PICKAXE)
        session.run(self.BUILD)
        self.assertEqual(self.presses(), ['hotbar.5', 'hotbar.1', 'hotbar.6'])
        self.assertEqual((session.block_slots[PICKAXE], session.block_slots['gold_block']), ('hotbar.6', 'hotbar.5'))

    def test_the_monitor_events_reach_the_timeline(self):
        from minecraftrl.build_sensor_repair import PICKAXE
        session = self.session(self.env([PICKAXE]), None)
        session.build.events.append(dict(event='anomaly', rule='stray', cell=[1, 64, 3], block='gold_block'))
        session.run(self.BUILD)
        lines = [json.loads(line) for line in (self.out / 'run/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        found = [e for e in lines if e['kind'] == 'repair_monitor']
        self.assertEqual([(e['event'], e['rule']) for e in found if e['event'] == 'anomaly'], [('anomaly', 'stray')])
        # the fake inventory holds none of the gifts: the ledger reports them once as unaccounted
        self.assertEqual([e['event'] for e in found if e['event'] != 'anomaly'], ['unaccounted'])


if __name__ == '__main__':
    unittest.main()
