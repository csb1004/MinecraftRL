"""The fast monitor on its own thread (2026-11-16; parallel monitoring memo 2026-10-25, "병렬 작업자 역할").

The game control loop stays the only owner of env.step, policy inference and the record. After each dispatch it
publishes the observation sample without blocking; this worker runs the current step's monitor (FastMonitor) on
every sample in publish order and hands the events back. The control loop takes them at its next boundary (before a
dispatch, and when a step ends) and decides: record, or hold. The worker never sends game input and never writes
the record, so STOP keeps going through the gate alone.

- Bounded: a full inbox drops the sample instead of blocking the control loop; drops are counted with their seq.
- Heartbeat: while a step is active the worker wakes at least every `heartbeat_s` and records one
  `observation_stale` event (recorded only) when no sample came for `stale_s` (e.g. env.step itself hangs).
- A monitor error stops the worker and is raised in the control thread at its next wait/poll, like the inline monitor.
"""
import threading
import time
from collections import deque

STALE_KIND = 'observation_stale'


class MonitorThread:
    def __init__(self, *, capacity=1024, heartbeat_s=.1, stale_s=5., clock=time.monotonic):
        self.capacity = capacity; self.heartbeat_s = heartbeat_s; self.stale_s = stale_s; self.clock = clock
        self._cv = threading.Condition(); self._inbox = deque(); self._outbox = []
        self._pending = 0  # samples waiting in the inbox
        self._published = 0; self._processed = 0; self._dropped = []; self._latency_ms = []
        self._heartbeats = 0; self._stale_events = 0; self._error = None; self._closed = False
        # Worker-only state: the active step's monitor, the last sample seen and whether its gap was reported.
        self._monitor = None; self._last = None; self._stale_reported = False
        self._thread = threading.Thread(target=self._run, name='oversight-monitor', daemon=True)

    # ---- control thread
    def start(self): self._thread.start(); return self

    def begin(self, monitor):
        """A policy step starts: later samples go to this monitor (an object with update(sample) -> event or None)."""
        self._put(('begin', monitor))

    def end(self):
        """The step ended: samples after this are counted but not monitored until the next begin()."""
        self._put(('end',))

    def publish(self, seq, sample):
        """Queue one sample without blocking; False when it was dropped (inbox full or thread closed)."""
        with self._cv:
            if self._closed or self._error is not None: return False
            if self._pending >= self.capacity:
                self._dropped.append(seq); return False
            self._inbox.append(('sample', seq, sample, self.clock())); self._pending += 1; self._published += 1
            self._cv.notify_all()
        return True

    def wait(self, timeout):
        """Wait at most `timeout` seconds until every published sample is processed; False if still behind."""
        with self._cv:
            done = self._cv.wait_for(lambda: self._error is not None or (self._processed >= self._published and not self._inbox),
                                     timeout=max(0., timeout))
            self._raise()
            return bool(done)

    def poll(self):
        """Events produced since the last poll, in sample order (never blocks)."""
        with self._cv:
            self._raise()
            events, self._outbox = self._outbox, []
        return events

    def close(self, timeout=1.):
        """Stop the worker; True when it has exited."""
        with self._cv:
            self._closed = True; self._cv.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread: self._thread.join(timeout)
        return not self._thread.is_alive()

    def alive(self): return self._thread.is_alive()

    def stats(self):
        with self._cv:
            latency = sorted(self._latency_ms)
            def pct(q): return latency[min(len(latency) - 1, int(q * len(latency)))] if latency else None
            return dict(published=self._published, processed=self._processed, dropped=len(self._dropped),
                        dropped_seq=list(self._dropped), capacity=self.capacity, heartbeat_s=self.heartbeat_s,
                        stale_s=self.stale_s, heartbeats=self._heartbeats, stale_events=self._stale_events,
                        latency_ms=dict(p50=pct(.5), p95=pct(.95), max=latency[-1] if latency else None),
                        error=None if self._error is None else type(self._error).__name__ + ': ' + str(self._error),
                        stopped=self._closed and not self._thread.is_alive())

    def _put(self, item):
        with self._cv:
            self._raise()
            self._inbox.append(item); self._cv.notify_all()

    def _raise(self):
        if self._error is not None:
            raise RuntimeError('monitor thread failed: %s: %s' % (type(self._error).__name__, self._error)) from self._error

    # ---- worker thread
    def _run(self):
        while True:
            with self._cv:
                while not self._inbox and not self._closed:
                    if not self._cv.wait(self.heartbeat_s): self._heartbeat()
                if self._closed and not self._inbox: return
                item = self._inbox.popleft()
                if item[0] == 'sample': self._pending -= 1
            try: events = self._handle(item)
            except BaseException as exc:
                with self._cv:
                    self._error = exc; self._cv.notify_all()
                return
            with self._cv:
                if item[0] == 'sample':
                    self._processed += 1; self._latency_ms.append((self.clock() - item[3]) * 1000.)
                self._outbox.extend(events); self._cv.notify_all()

    def _handle(self, item):
        if item[0] == 'begin':
            self._monitor, self._last, self._stale_reported = item[1], None, False; return []
        if item[0] == 'end':
            self._monitor, self._last = None, None; return []
        _, seq, sample, published = item
        if self._monitor is None: return []
        event = self._monitor.update(sample)
        self._last = (seq, getattr(sample, 'tick', None), self.clock()); self._stale_reported = False
        if not event: return []
        now = self.clock()
        return [dict(event, seq=seq, monitor_latency_ms=(now - published) * 1000., detected_mono=now)]

    def _heartbeat(self):
        """Under the lock, on a quiet wake-up: one stale-observation event per gap while a step is active."""
        self._heartbeats += 1
        if self._monitor is None or self._last is None or self._stale_reported: return
        seq, tick, seen = self._last; now = self.clock()
        if now - seen < self.stale_s: return
        self._stale_reported = True; self._stale_events += 1
        step = getattr(getattr(self._monitor, 'step', None), 'id', None) or getattr(self._monitor, 'name', None)
        self._outbox.append(dict(kind=STALE_KIND, hold=False, tick=tick, step=step, seq=seq, monitor_latency_ms=0.,
                                 detected_mono=now, evidence=dict(age_s=now - seen, stale_s=self.stale_s)))
