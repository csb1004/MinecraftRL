"""Entity tracks from entity-ray sightings by position continuity (sensor-tier design §2.6-3, §3.1; 2026-11-23)."""
import unittest

from minecraftrl.pipeline.entity_track import MATCH, EntityTracks


class EntityTrackTests(unittest.TestCase):
    def test_sightings_of_one_entity_join_one_track_and_a_far_one_starts_another(self):
        tracks = EntityTracks()
        # Two rays meet the same item in one frame: one observation at their mean point.
        tracks.update(1, [('item:beacon', (1.0, 64.1, 2.0), 2.), ('item:beacon', (1.2, 64.1, 2.0), 2.1)])
        self.assertEqual(len(tracks.tracks), 1)
        first = tracks.tracks[0]
        self.assertEqual((first['kind'], first['frame'], first['seen']), ('item:beacon', 1, 1))
        self.assertAlmostEqual(first['point'][0], 1.1)
        tracks.update(2, [('item:beacon', (1.4, 64.1, 2.0), 2.)])              # moved a little: same track
        tracks.update(3, [('item:beacon', (1.4 + MATCH + 1., 64.1, 2.0), 2.)])  # too far: another one
        tracks.update(3, [('cow', (1.4, 64.1, 2.0), 2.)])                       # another kind never joins
        self.assertEqual([(t['id'], t['kind'], t['seen']) for t in tracks.tracks],
                         [(1, 'item:beacon', 2), (2, 'item:beacon', 1), (3, 'cow', 1)])

    def test_latest_names_the_most_recent_track_of_a_kind_seen_since_a_frame(self):
        tracks = EntityTracks()
        tracks.update(5, [('item:gold_block', (0.5, 64.1, 3.5), 2.)])
        tracks.update(9, [('item:gold_block', (4.5, 64.1, 3.5), 2.)])
        self.assertEqual(tracks.latest('item:gold_block')['point'], (4.5, 64.1, 3.5))
        self.assertEqual(tracks.latest('item:gold_block', since=6)['frame'], 9)
        self.assertIsNone(tracks.latest('item:gold_block', since=10))
        self.assertIsNone(tracks.latest('item:beacon'))


if __name__ == '__main__':
    unittest.main()
