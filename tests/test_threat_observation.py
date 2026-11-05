import unittest
from minecraftrl.threat_observation import active_threat,active_pursuit,grounded


class ThreatObservationTests(unittest.TestCase):
    def test_ground_state_must_be_valid_and_settled(self):
        self.assertTrue(grounded(dict(server_on_ground=1,server_fall_distance=0.)))
        self.assertFalse(grounded(dict(server_on_ground=0,server_fall_distance=0.)))
        self.assertFalse(grounded(dict(server_on_ground=1,server_fall_distance=1.)))
        for value in ({},dict(server_on_ground=2,server_fall_distance=0.),dict(server_on_ground=1,server_fall_distance=float('nan'))):
            with self.assertRaises(ValueError):grounded(value)

    def test_positive_and_explicit_empty_scan(self):
        self.assertTrue(active_threat(dict(threat_valid=1,threat_scan_radius=24,threat_count=2,threat_distance=11.5)))
        self.assertFalse(active_threat(dict(threat_valid=1,threat_scan_radius=24,threat_count=0,threat_distance=-1)))

    def test_missing_or_inconsistent_scan_cannot_mean_safe(self):
        base=dict(threat_valid=1,threat_scan_radius=24,threat_count=0,threat_distance=-1)
        for key,value in [('threat_valid',0),('threat_count',.5),('threat_distance',3),
                          ('threat_scan_radius',16),('threat_distance',float('nan'))]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                active_threat(dict(base,**{key:value}))
        with self.assertRaises(ValueError):active_threat({})

    def test_occluded_pursuer_is_positive_without_visible_threat(self):
        scan=dict(threat_valid=1,threat_scan_radius=24,threat_count=0,threat_distance=-1,
            pursuit_valid=1,pursuit_scan_radius=32,pursuit_count=1,pursuit_distance=25.)
        self.assertFalse(active_threat(scan));self.assertTrue(active_pursuit(scan))
        self.assertFalse(active_pursuit(dict(scan,pursuit_count=0,pursuit_distance=-1)))
        for change in (dict(pursuit_valid=0),dict(pursuit_distance=33),dict(pursuit_scan_radius=24),
                       dict(threat_count=2,threat_distance=12),dict(threat_count=1,threat_distance=12)):
            with self.subTest(change=change),self.assertRaises(ValueError):active_pursuit(dict(scan,**change))
