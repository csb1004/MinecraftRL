import unittest

from minecraftrl.stall_detection import StallDetector, StallSample, observed_cycles


def sample(tick, x=0.0, stage='approach', **kw):
    return StallSample(tick=tick, position=(x, 64., 0.), yaw=0., stage=stage,
                       action=('attack',), **kw)


class StallDetectionTests(unittest.TestCase):
    def test_stationary_attack_emits_once(self):
        d=StallDetector(); rows=[d.update(sample(i)) for i in range(450)]
        alerts=[r for r in rows if r['event']=='stalled']
        self.assertEqual(len(alerts),1)
        self.assertEqual(alerts[0]['tick'],200)
        self.assertEqual(alerts[0]['reason'],'stationary_repetition')

    def test_mining_progress_allows_repeated_attack(self):
        d=StallDetector()
        for i in range(600):
            r=d.update(sample(i,stage='mine',target=(0,64,1),mining_progress=i/600.))
            self.assertFalse(r['stalled'])

    def test_mining_flag_without_damage_does_not_exempt_forever(self):
        d=StallDetector()
        rows=[d.update(sample(i,stage='mine',target=(0,64,1))) for i in range(450)]
        self.assertTrue(any(r['event']=='stalled' for r in rows))

    def test_moving_away_on_new_route_is_progress(self):
        d=StallDetector()
        for i in range(500):
            r=d.update(sample(i,x=i*.03,target=(0,64,1),distance=5+i*.03))
            self.assertFalse(r['stalled'])

    def test_repeated_walk_cycle_detected(self):
        d=StallDetector(); alerts=[]
        for i in range(450):
            r=d.update(sample(i,x=(0.,2.,4.,2.)[(i//20)%4]))
            if r['event']=='stalled':alerts.append(r)
        self.assertEqual(len(alerts),1)
        self.assertEqual(alerts[0]['reason'],'repeated_cycle')
        self.assertGreaterEqual(alerts[0]['cycle']['count'],2)

    def test_single_backtrack_then_new_route_is_allowed(self):
        d=StallDetector()
        for i in range(400):
            x=i*.04 if i<50 else (100-i)*.04 if i<100 else -(i-100)*.04
            self.assertFalse(d.update(sample(i,x=x))['stalled'])

    def test_target_stage_oscillation_cannot_reset_forever(self):
        d=StallDetector()
        rows=[d.update(sample(i,stage='search' if i%2 else 'approach',target=(i%2,64,1)))
              for i in range(500)]
        self.assertTrue(any(r['event']=='stalled' for r in rows))

    def test_position_boundary_jitter_eventually_stalls(self):
        d=StallDetector()
        rows=[d.update(sample(i,x=-.001 if i%2 else .001)) for i in range(450)]
        self.assertTrue(any(r['event']=='stalled' for r in rows))

    def test_cycle_witnesses_are_chronological_and_directional(self):
        cycles=observed_cycles(['a','b','c','a','b','c','a'])
        self.assertEqual(cycles[0]['count'],2)
        self.assertEqual(cycles[0]['nodes'],['a','b','c','a'])
        self.assertEqual(observed_cycles(['a','b','c','d']),[])

    def test_stale_tick_is_not_policy_stall(self):
        d=StallDetector(); d.update(sample(10))
        r=d.update(sample(10))
        self.assertEqual(r['event'],'observation_gap'); self.assertFalse(r['stalled'])
        self.assertFalse(d.update(sample(11))['stalled'])

    def test_large_tick_gap_is_not_observed_stationarity(self):
        d=StallDetector(); d.update(sample(0))
        self.assertEqual(d.update(sample(500))['event'],'observation_gap')

    def test_terminal_and_stop_prevent_further_updates(self):
        for terminal in (True,False):
            d=StallDetector()
            if terminal:d.update(sample(0,terminal=True))
            else:d.stop()
            with self.assertRaises(RuntimeError):d.update(sample(1))

    def test_inventory_high_water_not_drop_pickup_loop(self):
        d=StallDetector()
        rows=[d.update(sample(i,inventory_count=i%2)) for i in range(450)]
        self.assertTrue(any(r['event']=='stalled' for r in rows))

    def test_invalid_numeric_observation_rejected(self):
        with self.assertRaises(ValueError):StallDetector().update(sample(0,x=float('nan')))

    def test_reversed_directed_cycle_is_not_the_same_attempt(self):
        cycles=observed_cycles(['a','b','c','a','c','b','a'])
        self.assertEqual(len(cycles),2)
        self.assertEqual([c['count'] for c in cycles],[1,1])

    def test_real_progress_releases_one_alarm_episode(self):
        d=StallDetector()
        for i in range(201):d.update(sample(i,target=(8,64,0),distance=8.))
        r=d.update(sample(201,x=1.,target=(8,64,0),distance=7.))
        self.assertEqual(r['event'],'progress_resumed')
        self.assertFalse(r['stalled'])

    def test_damage_reset_does_not_create_unlimited_progress(self):
        d=StallDetector()
        rows=[d.update(sample(i,stage='mine',target=(0,64,1),mining_progress=(i%10)/10.))
              for i in range(500)]
        self.assertTrue(any(r['event']=='stalled' for r in rows))

    def test_new_cells_without_goal_evidence_have_finite_budget(self):
        d=StallDetector()
        rows=[d.update(sample(i,x=i*.1,stage='search')) for i in range(900)]
        alerts=[r for r in rows if r['event']=='stalled']
        self.assertEqual(len(alerts),1)
        self.assertEqual(alerts[0]['reason'],'no_goal_progress')

    def test_different_world_or_goal_revision_cannot_mix_history(self):
        d=StallDetector();d.update(sample(0,context_id='world1/task1/r1'))
        with self.assertRaises(ValueError):d.update(sample(1,context_id='world1/task1/r2'))

    def test_per_episode_memory_has_explicit_bound(self):
        d=StallDetector();d.max_samples=2
        d.update(sample(0));d.update(sample(1))
        with self.assertRaises(RuntimeError):d.update(sample(2))


if __name__=='__main__':unittest.main()
