"""Linux ownership tracking for MineRL children which start separate sessions.

A supervisor becomes a child subreaper before launch. Thus double-forked/setsid
Java descendants remain attributable after worker exit. PID + /proc start time
protects against signaling a recycled PID. Pre-existing children are excluded.
"""
import ctypes
import os
from pathlib import Path
import signal
import time


def snapshot():
    table = {}
    # Do not glob nested paths: pathlib may stat them outside our exception
    # boundary while a process exits. Enumerate names, then read inside try.
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        path = folder / 'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            pid = int(path.parent.name)
            table[pid] = {'ppid': int(fields[1]), 'start': int(fields[19]), 'state': fields[0]}
        except (OSError, ValueError, IndexError): continue
    return table


def descendants(table, root):
    found = set(); frontier = {root}
    while frontier:
        children = {pid for pid, info in table.items() if info['ppid'] in frontier and pid not in found}
        found.update(children); frontier = children
    return found


class ProcessOwner:
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        old = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(old), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER failed')
        self.previous_subreaper = old.value
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
        self.parent = os.getpid(); table = snapshot()
        self.baseline = {(pid, table[pid]['start']) for pid in descendants(table, self.parent)}
        self.owned = set()

    def capture(self):
        table = snapshot()
        for pid in descendants(table, self.parent):
            identity = (pid, table[pid]['start'])
            if identity not in self.baseline: self.owned.add(identity)
        return table

    def live(self):
        table = self.capture()
        return {(pid, start) for pid, start in self.owned if pid in table and table[pid]['start'] == start
                and table[pid]['state'] != 'Z'}

    def send(self, sig):
        for pid, start in self.live():
            # Recheck immediately before signaling; never signal by name/global JVM.
            table = snapshot()
            if pid in table and table[pid]['start'] == start:
                try: os.kill(pid, sig)
                except ProcessLookupError: pass

    def close(self, worker):
        try:
            self.send(signal.SIGTERM)
            deadline = time.monotonic() + 1.
            while self.live() and time.monotonic() < deadline: time.sleep(.02)
            self.send(signal.SIGKILL)
            try: worker.wait(timeout=3)
            except Exception: pass
            deadline = time.monotonic() + 1.
            while self.live() and time.monotonic() < deadline: time.sleep(.02)
            # Reap adopted grandchildren, leaving worker's status to Popen.wait.
            for pid, start in self.owned:
                if pid == worker.pid: continue
                try: os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, ProcessLookupError): pass
            remaining = sorted(pid for pid, _ in self.live())
            return {'owned_processes_seen': len(self.owned), 'remaining_live_pids': remaining,
                    'method': 'Linux subreaper + descendant PID/start-time ownership'}
        finally:
            self.libc.prctl(36, self.previous_subreaper, 0, 0, 0)
