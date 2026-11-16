import threading
import time
import unittest

from minecraftrl.pipeline.monitor_thread import MonitorThread


class Seen:
    """Monitor stand-in: records each sample and returns an event at the samples it is told to."""
    def __init__(self, name='s', at=(), hold=False):
        self.name = name; self.at = set(at); self.hold = hold; self.samples = []; self.threads = set()
    def update(self, sample):
        self.samples.append(sample); self.threads.add(threading.get_ident())
        if sample in self.at: return dict(kind='probe', hold=self.hold, tick=sample, step=self.name, evidence={})
        return None


class Blocking(Seen):
    """Blocks inside update until released; `entered` is set while it waits."""
    def __init__(self, **kw):
        super().__init__(**kw); self.entered = threading.Event(); self.release = threading.Event()
    def update(self, sample):
        self.entered.set(); self.release.wait(5.)
        return super().update(sample)


class Failing(Seen):
    def update(self, sample): raise ValueError('bad sample %s' % sample)


def until(predicate, timeout=2.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate(): return True
        time.sleep(.002)
    return False


class MonitorThreadTests(unittest.TestCase):
    def setUp(self):
        self.thread = MonitorThread(); self.thread.start(); self.addCleanup(self.thread.close)

    def test_every_published_sample_is_processed_in_order_off_the_control_thread(self):
        monitor = Seen(); self.thread.begin(monitor)
        for seq in range(1, 201): self.assertTrue(self.thread.publish(seq, seq))
        self.assertTrue(self.thread.wait(2.))
        self.assertEqual(monitor.samples, list(range(1, 201)))
        self.assertNotIn(threading.get_ident(), monitor.threads)
        stats = self.thread.stats()
        self.assertEqual((stats['published'], stats['processed'], stats['dropped']), (200, 200, 0))

    def test_an_event_carries_the_seq_of_its_sample(self):
        self.thread.begin(Seen('iron', at={5}, hold=True))
        for seq in range(1, 9): self.thread.publish(seq, seq)
        self.thread.wait(2.)
        events = self.thread.poll()
        self.assertEqual([(e['kind'], e['hold'], e['step'], e['seq']) for e in events], [('probe', True, 'iron', 5)])
        self.assertGreaterEqual(events[0]['monitor_latency_ms'], 0.)
        self.assertIn('detected_mono', events[0])
        self.assertEqual(self.thread.poll(), [])  # taken once

    def test_steps_switch_monitors_in_publish_order(self):
        first, second = Seen('a'), Seen('b')
        self.thread.begin(first); self.thread.publish(1, 1); self.thread.publish(2, 2); self.thread.end()
        self.thread.begin(second); self.thread.publish(3, 3)
        self.thread.wait(2.)
        self.assertEqual((first.samples, second.samples), ([1, 2], [3]))

    def test_samples_outside_a_step_are_not_monitored(self):
        self.thread.publish(1, 1); self.thread.wait(2.)
        self.assertEqual(self.thread.stats()['processed'], 1)  # counted, nothing to update

    def test_wait_is_bounded_while_the_monitor_lags(self):
        monitor = Blocking(); self.thread.begin(monitor); self.thread.publish(1, 1)
        self.assertTrue(monitor.entered.wait(2.))
        started = time.monotonic()
        self.assertFalse(self.thread.wait(.02))
        self.assertLess(time.monotonic() - started, 1.)
        monitor.release.set()
        self.assertTrue(self.thread.wait(2.))

    def test_a_full_inbox_drops_without_blocking_and_counts(self):
        thread = MonitorThread(capacity=2); thread.start(); self.addCleanup(thread.close)
        monitor = Blocking(); thread.begin(monitor)
        self.assertTrue(thread.publish(1, 1)); self.assertTrue(monitor.entered.wait(2.))  # 1 is being processed
        self.assertEqual([thread.publish(seq, seq) for seq in (2, 3, 4, 5)], [True, True, False, False])
        monitor.release.set(); self.assertTrue(thread.wait(2.))
        stats = thread.stats()
        self.assertEqual((stats['published'], stats['processed'], stats['dropped'], stats['dropped_seq']), (3, 3, 2, [4, 5]))
        self.assertEqual(monitor.samples, [1, 2, 3])

    def test_a_monitor_error_reaches_the_control_thread(self):
        self.thread.begin(Failing()); self.thread.publish(1, 1)
        self.assertTrue(until(lambda: self.thread.stats()['error'] is not None))
        with self.assertRaisesRegex(RuntimeError, 'bad sample 1'): self.thread.poll()
        with self.assertRaisesRegex(RuntimeError, 'bad sample 1'): self.thread.wait(.1)

    def test_heartbeat_records_one_stale_observation_per_gap(self):
        now = [0.]
        thread = MonitorThread(heartbeat_s=.002, stale_s=5., clock=lambda: now[0]); thread.start(); self.addCleanup(thread.close)
        thread.begin(Seen('iron')); thread.publish(7, 7); thread.wait(2.)
        now[0] = 6.
        self.assertTrue(until(lambda: thread.stats()['stale_events'] == 1))
        time.sleep(.02)  # more heartbeats in the same gap
        events = thread.poll()
        self.assertEqual([(e['kind'], e['hold'], e['step'], e['seq']) for e in events], [('observation_stale', False, 'iron', 7)])
        self.assertGreaterEqual(events[0]['evidence']['age_s'], 5.)
        thread.publish(8, 8); thread.wait(2.); now[0] = 8.; time.sleep(.02)
        self.assertEqual(thread.poll(), [])  # a new sample starts a new gap
        thread.end(); now[0] = 100.; time.sleep(.02)
        self.assertEqual(thread.poll(), [])  # no step, no stale check
        self.assertGreater(thread.stats()['heartbeats'], 0)

    def test_close_stops_the_worker(self):
        thread = MonitorThread(); thread.start()
        self.assertTrue(thread.close(1.))
        self.assertFalse(thread.alive())
        self.assertFalse(thread.publish(1, 1))
        self.assertTrue(thread.stats()['stopped'])


if __name__ == '__main__':
    unittest.main()
