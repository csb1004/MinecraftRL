"""Append-only run record: timeline events, per-step telemetry, and frames for the replay."""
import json
import time
from pathlib import Path

from ..runtime import jsonable


class Recorder:
    def __init__(self, output, *, clock=time.monotonic, frame_every=1, quality=80):
        self.output = Path(output); (self.output / 'frames').mkdir(parents=True, exist_ok=True)
        self.clock = clock; self.frame_every = frame_every; self.quality = quality; self.step = 0
        self.timeline = (self.output / 'timeline.jsonl').open('a', encoding='utf-8')
        self.samples = (self.output / 'telemetry.jsonl').open('a', encoding='utf-8')

    def write(self, stream, value):
        stream.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()

    def event(self, kind, **fields):
        step = fields.pop('step', self.step)
        self.write(self.timeline, {'t': self.clock(), 'step': step, 'kind': kind, **fields})

    def telemetry(self, step, sample):
        self.step = step; self.write(self.samples, {'t': self.clock(), 'step': step, **sample})

    def frame(self, step, rgb):
        self.step = step
        if step % self.frame_every: return
        from PIL import Image
        Image.fromarray(rgb).save(self.output / 'frames' / ('%06d.jpg' % step), quality=self.quality)

    def close(self):
        self.timeline.close(); self.samples.close()
