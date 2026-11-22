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


if __name__ == '__main__':
    unittest.main()
