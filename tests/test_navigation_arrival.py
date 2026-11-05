import unittest
from minecraftrl.navigation_arrival import ArrivalWindow


def pose(x=0., y=201., tick=1, screen=0):
    return dict(server_x=x, server_y=y, server_z=0., yaw=0., pitch=0.,
                server_tick=tick, server_available=1, error=0, screen=screen)


class ArrivalWindowTests(unittest.TestCase):
    target = [0., 201., 0.]

    def test_legacy_radius_entry_remains_instantaneous(self):
        self.assertTrue(ArrivalWindow().advance(pose(x=.6), pose(x=.3, tick=2), self.target))

    def test_fast_crossing_cannot_start_a_settled_hold(self):
        window = ArrivalWindow(12)
        self.assertFalse(window.advance(pose(x=.6), pose(x=.3, tick=2), self.target))
        self.assertEqual(window.count, 0)
        for tick in range(2, 13):
            self.assertFalse(window.advance(pose(tick=tick), pose(tick=tick+1), self.target))
        self.assertTrue(window.advance(pose(tick=13), pose(tick=14), self.target))

    def test_exit_motion_height_and_gui_reset_contiguous_evidence(self):
        for bad in (pose(x=.4, tick=3), pose(x=.1, tick=3), pose(y=201.1, tick=3),
                    pose(screen=1, tick=3)):
            window = ArrivalWindow(3)
            window.advance(pose(), pose(tick=2), self.target)
            self.assertFalse(window.advance(pose(tick=2), bad, self.target))
            self.assertEqual(window.count, 0)

    def test_new_target_or_owner_cannot_inherit_hold_evidence(self):
        window = ArrivalWindow(2)
        window.advance(pose(), pose(tick=2), self.target)
        window.reset()
        self.assertFalse(window.advance(pose(tick=2), pose(tick=3), self.target))

    def test_invalid_duration_or_missing_tick_evidence_rejected(self):
        for duration in (0, 21, 1.0, True):
            with self.assertRaises(ValueError): ArrivalWindow(duration)
        with self.assertRaises(ValueError):
            ArrivalWindow(12).advance(pose(), pose(tick=3), self.target)
