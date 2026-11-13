"""Live skill stages for the reacquire head, and the record of policy latents (design 2026-11-07 §3, §5, §6).

LiveStageFeed gets every recorded telemetry row (the same dict the recorder writes) in order and keeps the shared
StageTracker of stage_replay, so live play and the offline dataset builder judge stages with one function.
Fixture positions and the server pose are privileged observations: the stage input is disclosed as such.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .monitor_replay import sample_from_row
from .stage_replay import STAGES, Geometry, StageTracker, fixture_blocks, fixture_origin

LATENT_SCHEMA = 'reacquire_latents_v1'


class LiveStageFeed:
    def __init__(self, fixture, wanted, target_item):
        self.fixture = fixture; self.wanted = wanted; self.target_item = target_item
        self.base = None           # fixture geometry once the first applied row gives the origin
        self.events = []           # (t, position, block) of every break since the run started
        self.prev_destroys = None
        self.tracker = None; self.applied = 0
        self.last_row = None; self.last_sample = None; self.last_view = None

    def observe(self, row):
        """One recorded telemetry row; policy rows of the current step advance the tracker. A row the sample adapter
        cannot read (a field missing before server telemetry is live) is an unknown stage, never a crash of the run."""
        try: s = sample_from_row(row)
        except (TypeError, ValueError, KeyError):
            self.last_row, self.last_sample, self.last_view = row, None, None
            return
        if self.base is None:
            start = fixture_origin([row])
            if start is not None: self.base = Geometry(fixture_blocks(self.fixture, *start), start[0][1])
        if s.valid:
            # Same rule as stage_replay.destroy_events: the destroy counter rose between valid rows.
            if self.prev_destroys is not None and s.destroys > self.prev_destroys and s.last_destroyed_pos is not None:
                self.events.append((row['t'], s.last_destroyed_pos, s.last_destroyed))
            self.prev_destroys = s.destroys
        self.last_row, self.last_sample, self.last_view = row, s, None
        if row.get('controller') == 'policy' and s.valid and self._tracker() is not None:
            self._apply_due(s)
            self.last_view = self.tracker.update(s)

    def start_step(self, start_t):
        """A new policy step: a fresh tracker over the run-start geometry (all breaks so far are applied again)."""
        self.start_t = start_t; self.tracker = None; self.applied = 0

    def _tracker(self):
        if self.tracker is None and self.base is not None and hasattr(self, 'start_t'):
            self.tracker = StageTracker(self.base.copy(), self.wanted, self.target_item, self.start_t)
        return self.tracker

    def _apply_due(self, s):
        due = [e for e in self.events[self.applied:] if e[0] <= self.last_row['t']]
        self.applied += len(due)
        self.tracker.apply_destroys(due, s)

    def decision_view(self):
        """Stage of the state the next policy action will see: the last policy row's view, or for the first decision
        after a non-policy row a peek at that row. None while the stage is unknown (no fixture origin or invalid row)."""
        if self.last_sample is None or not self.last_sample.valid or self._tracker() is None: return None
        if self.last_view is not None: return self.last_view
        self._apply_due(self.last_sample)     # breaks before the step only remove blocks (no collect before start)
        return self.tracker.peek(self.last_sample)

    def features(self):
        """Privileged direction features of the decision view (mining direction input design 2026-11-14 §1)."""
        from ..reacquire_teacher import mining_features
        return mining_features(self.last_sample, self.decision_view(), self.wanted)


class LatentLog:
    """Policy latents per dispatched policy step: chunked npz files and a manifest kept current after each chunk."""

    def __init__(self, folder, *, width, cond_scale, meta, chunk=256, stages=STAGES, features=()):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True); self.stages = tuple(stages)
        self.width = int(width); self.cond_scale = float(cond_scale); self.meta = dict(meta); self.chunk = int(chunk)
        self.features = tuple(features)    # feature names the head saw (jump direction input, design 2026-11-14)
        self.rows = []; self.chunks = []; self.total = 0

    def add(self, step, latent, stage, features=None):
        value = latent.detach().float().cpu().numpy() if hasattr(latent, 'detach') else np.asarray(latent, dtype=np.float32)
        if value.shape != (2, 1, self.width): raise ValueError('latent must be [2, 1, width] (conditional, unconditional)')
        feats = np.zeros(len(self.features), np.float32) if features is None else np.asarray(features, np.float32)
        if feats.shape != (len(self.features),): raise ValueError('expected %d feature values' % len(self.features))
        self.rows.append((int(step), value[0, 0].astype(np.float16), value[1, 0].astype(np.float16),
                          -1 if stage is None else int(stage), feats))
        if len(self.rows) >= self.chunk: self._flush()

    def _flush(self):
        if not self.rows: return
        name = 'latents-%03d.npz' % len(self.chunks); path = self.folder / name
        steps, cond, uncond, stage, feats = zip(*self.rows)
        extra = dict(features=np.stack(feats)) if self.features else {}
        np.savez(path, step=np.array(steps, np.int32), cond=np.stack(cond), uncond=np.stack(uncond),
                 stage=np.array(stage, np.int8), **extra)
        self.chunks.append(dict(file=name, rows=len(self.rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.total += len(self.rows); self.rows = []
        self._manifest(complete=False)

    def _manifest(self, *, complete):
        manifest = dict(schema=LATENT_SCHEMA, width=self.width, cond_scale=self.cond_scale, stages=list(self.stages),
                        rows=self.total, chunks=list(self.chunks), complete=complete, meta=self.meta)
        if self.features: manifest['features'] = list(self.features)
        (self.folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return manifest

    def close(self):
        self._flush()
        return self._manifest(complete=True)


class ReacquireHook:
    """What DemoSession calls around policy actions of a skill with a `reacquire` block."""

    def __init__(self, head, fixture, wanted, target_item, folder, *, cond_scale, meta, chunk=256, feed=None):
        """`feed` replaces the mining stage feed for a skill with its own stages (jump course, design 2026-11-13); its
        views name stages of the head's stage list."""
        self.head = head; self.meta = dict(meta); self.decisions = 0
        self.stages = tuple(getattr(head, 'stages', STAGES))
        # A head with privileged features (jump direction input, design 2026-11-14) gets them from the feed each decision.
        self.features = tuple(getattr(head, 'features', ()))
        self.feed = feed if feed is not None else LiveStageFeed(fixture, wanted, target_item)
        self.log = None if folder is None else LatentLog(folder, width=head.width, cond_scale=cond_scale, meta=meta,
                                                          chunk=chunk, stages=self.stages, features=self.features)

    def observe(self, row): self.feed.observe(row)

    def activate(self, on):
        """Residual on for a policy step of the reacquire skill, off (official head, no stage) for any other."""
        self.head.active = bool(on)
        if not on: self.head.stage = None; self.head.feature = None

    def start_step(self, start_t): self.feed.start_step(start_t)

    def before_act(self):
        view = self.feed.decision_view()
        self.head.stage = None if view is None else self.stages.index(view.stage)
        if self.features: self.head.feature = list(self.feed.features())

    def after_dispatch(self, step):
        self.decisions += 1
        if self.log is not None:
            self.log.add(step, self.head.last_latent, self.head.stage, self.head.feature if self.features else None)

    def close(self):
        manifest = self.log.close() if self.log is not None else None
        return dict(self.meta, decisions=self.decisions, latent_rows=manifest['rows'] if manifest else 0)
