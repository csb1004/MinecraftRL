import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.frame_labels import CLASSES, balanced_sample, block_class, labelled_frames


def telemetry_row(step, hit='air', screen=0., available=1.):
    return {'t': step / 20., 'step': step, 'screen': screen, 'server_available': available, 'controller': 'policy',
            'demo': {'demo_available': 1., 'demo_hit_block': hit}}


class FrameLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'R1'
        (self.run / 'frames').mkdir(parents=True)
        for step in (2, 4, 6, 8, 10):
            (self.run / 'frames' / f'{step:06d}.jpg').write_bytes(b'')
        rows = [telemetry_row(2, 'oak_log'), telemetry_row(4, 'iron_ore'), telemetry_row(6, 'lapis_ore'),
                telemetry_row(8, 'stone', screen=1.), telemetry_row(9, 'dirt')]
        (self.run / 'telemetry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    def test_block_classes(self):
        self.assertEqual([block_class(b) for b in ('birch_log', 'iron_ore', 'barrier', 'gravel', 'air')],
                         ['log', 'iron_ore', 'barrier', 'other', 'air'])
        self.assertIn('other', CLASSES)

    def test_frames_join_rows_by_step_and_skip_gui_and_unmatched(self):
        items = labelled_frames(self.run)
        self.assertEqual([(i['step'], i['label']) for i in items], [(2, 'log'), (4, 'iron_ore'), (6, 'other')])
        self.assertEqual(items[0]['path'], self.run / 'frames' / '000002.jpg')
        self.assertEqual(items[0]['run'], 'R1')

    def test_balanced_sample_caps_each_class_and_is_deterministic(self):
        items = [dict(label='stone', step=i) for i in range(10)] + [dict(label='iron_ore', step=i) for i in range(3)]
        a = balanced_sample(items, 4, seed=7)
        self.assertEqual(sorted((i['label'] for i in a)), ['iron_ore'] * 3 + ['stone'] * 4)
        self.assertEqual(a, balanced_sample(items, 4, seed=7))


if __name__ == '__main__':
    unittest.main()
