import tempfile
import unittest
from pathlib import Path
from minecraftrl.runtime import run_episode, StopGate, inventory_counts, quantile


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t


class Environment:
    def __init__(self, missing=False):
        self.actions = []; self.released = 0; self.closed = False; self.seeds = []; self.missing = missing
    def seed(self, seed): self.seeds.append(seed)
    def reset(self):
        assert self.seeds, 'seed must precede reset'
        return {} if self.missing else {'pov': 'frame', 'inventory': {'oak_log': 2, 'dirt': 0}}
    def step(self, action):
        self.actions.append(action)
        return {'pov': 'frame', 'inventory': {'oak_log': 3, 'dirt': 0}}, 0, len(self.actions) == 2, {}
    def release(self): self.released += 1
    def close(self): self.closed = True


class Policy:
    def __init__(self, clock, during=None): self.clock = clock; self.during = during; self.resets = []; self.closed = False
    def reset(self, goal=None): self.resets.append(goal)
    def act(self, screen):
        self.clock.t += .1
        if self.during: self.during()
        return {'forward': 1, 'camera': [0, 0]}
    def close(self): self.closed = True


class RuntimeTests(unittest.TestCase):
    def test_stop_between_capture_and_policy_guard_opens_closed_control_archive(self):
        clock=Clock();env=Environment();policy=Policy(clock);calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            stop=Path(tmp)/'STOP'
            def capture(*args):stop.touch()
            def guarded_act(screen):
                if stop.exists():raise InterruptedError('STOP before inference')
                self.fail('expected STOP at policy boundary')
            policy.act=guarded_act
            report=run_episode(policy,env,tmp,seed=1,goal=None,clock=clock,stop_file=stop,
                save_screen=capture,on_control_closed=lambda report:calls.append(report['status']))
        self.assertEqual(report['status'],'cancelled');self.assertNotIn('error',report)
        self.assertFalse(report['execution_valid']);self.assertEqual(env.actions,[])
        self.assertEqual(calls,['cancelled']);self.assertTrue(policy.closed);self.assertTrue(env.closed)

    def test_unrequested_interrupted_error_does_not_open_archive_grace(self):
        clock=Clock();env=Environment();policy=Policy(clock);calls=[]
        def failure(screen):raise InterruptedError('unexpected backend interruption')
        policy.act=failure
        with tempfile.TemporaryDirectory() as tmp:
            report=run_episode(policy,env,tmp,seed=1,goal=None,clock=clock,
                save_screen=lambda *a:None,on_control_closed=lambda report:calls.append(1))
        self.assertEqual(report['status'],'error');self.assertIn('InterruptedError',report['error'])
        self.assertEqual(calls,[]);self.assertEqual(env.actions,[])

    def test_archive_boundary_follows_release_and_policy_close_before_slow_environment_close(self):
        clock=Clock();env=Environment();policy=Policy(clock);calls=[]
        def boundary(report):
            self.assertEqual(env.released,1);self.assertTrue(policy.closed);self.assertFalse(env.closed)
            calls.append(report['status'])
        with tempfile.TemporaryDirectory() as tmp:
            run_episode(policy,env,tmp,seed=1,goal=None,clock=clock,save_screen=lambda *a:None,
                        on_control_closed=boundary)
        self.assertEqual(calls,['environment_done']);self.assertTrue(env.closed)

    def test_failed_input_release_never_opens_archive_grace(self):
        clock=Clock();env=Environment();policy=Policy(clock);calls=[]
        def fail():raise RuntimeError('release failed')
        env.release=fail
        with tempfile.TemporaryDirectory() as tmp:
            run_episode(policy,env,tmp,seed=1,goal=None,clock=clock,save_screen=lambda *a:None,
                        on_control_closed=lambda report:calls.append(1))
        self.assertEqual(calls,[]);self.assertTrue(env.closed);self.assertTrue(policy.closed)
    def run_case(self, **kwargs):
        clock = Clock(); env = Environment(kwargs.pop('missing', False)); gate = StopGate()
        during = kwargs.pop('during', None)
        policy = Policy(clock, (lambda: during(gate, clock)) if during else None)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_episode(policy, env, Path(tmp), seed=104729, goal=None,
                target_items=['oak_log'], limit_s=kwargs.pop('limit_s', 180), gate=gate,
                clock=clock, save_screen=lambda frame, path: path.write_text('test frame'), **kwargs)
            self.assertTrue((Path(tmp)/'summary.json').exists())
        self.assertTrue(env.closed); self.assertTrue(policy.closed); self.assertEqual(env.released, 1)
        return report, env, policy

    def test_stop_during_inference_drops_action(self):
        report, env, _ = self.run_case(during=lambda gate, clock: gate.stop('test_cancel'))
        self.assertEqual(env.actions, [])
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['steps'], 0)

    def test_deadline_during_inference_drops_action(self):
        report, env, _ = self.run_case(limit_s=.05)
        self.assertEqual(env.actions, [])
        self.assertEqual(report['status'], 'time_limit')
        self.assertFalse(report['execution_valid'])

    def test_inventory_proves_acquisition_and_resets_policy(self):
        report, env, policy = self.run_case()
        self.assertEqual(policy.resets, [None])
        self.assertEqual(env.seeds, [104729])
        self.assertEqual(report['inventory_peak_gain']['oak_log'], 1)
        self.assertTrue(report['target_observed'])
        self.assertEqual(report['steps'], 2)
        self.assertTrue(report['execution_valid'])
        self.assertAlmostEqual(report['inference_ms']['p95'], 100.)

    def test_missing_inventory_is_failure_not_zero_success(self):
        report, env, _ = self.run_case(missing=True)
        self.assertEqual(report['status'], 'error')
        self.assertIn('inventory', report['error'])
        self.assertFalse(report['execution_valid'])
        self.assertEqual(env.actions, [])

    def test_initial_inventory_is_not_acquisition(self):
        report, _, _ = self.run_case(target_count=2)
        self.assertFalse(report['target_observed'])

    def test_control_duration_excludes_cleanup(self):
        clock = Clock(); env = Environment(); policy = Policy(clock)
        def slow_close():
            clock.t += 12
        env.close = slow_close
        with tempfile.TemporaryDirectory() as tmp:
            report = run_episode(policy, env, tmp, seed=1, goal=None, clock=clock,
                                 save_screen=lambda *_: None)
        self.assertAlmostEqual(report['elapsed_control_s'], .2)
        self.assertGreaterEqual(report['elapsed_cleanup_s'], 12)

    def test_failure_releases_and_records_traceback(self):
        def explode(gate, clock): raise RuntimeError('policy broke')
        report, env, _ = self.run_case(during=explode)
        self.assertEqual(report['status'], 'error')
        self.assertIn('policy broke', report['error'])
        self.assertEqual(env.actions, [])

    def test_stop_is_latched_and_never_dispatches_again(self):
        gate = StopGate(); actions = []
        gate.stop('cancel')
        self.assertFalse(gate.dispatch(lambda: actions.append('late'))[0])
        self.assertEqual(actions, [])
        self.assertEqual(gate.reason, 'cancel')

    def test_inventory_validation_and_quantiles(self):
        self.assertEqual(inventory_counts({'inventory': {'dirt': 3}}), {'dirt': 3})
        with self.assertRaises(ValueError): inventory_counts({'inventory': {'dirt': -1}})
        with self.assertRaises(ValueError): inventory_counts({'inventory': {'dirt': 1.5}})
        self.assertIsNone(quantile([], .95))
        self.assertEqual(quantile([1, 2, 3, 4, 5], .5), 3)

if __name__ == '__main__': unittest.main()
