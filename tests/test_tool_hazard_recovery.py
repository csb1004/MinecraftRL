import copy
import unittest
from unittest.mock import Mock

from minecraftrl.runtime import StopGate
from minecraftrl.tool_collection import ToolCollectionEnvironment, make_ledger, validate_condition_phase, health_limits_for_phase
from minecraftrl.threat_observation import PursuitStability


def server(tick, **changes):
    result = dict(server_tick=tick, server_available=1, error=0, screen=0,
        server_alive=1, server_health=2.166666, server_on_ground=1, server_fall_distance=0,
        threat_valid=1, threat_scan_radius=24, threat_count=0, threat_distance=-1,
        pursuit_valid=1, pursuit_scan_radius=32, pursuit_count=0, pursuit_distance=-1,
        server_selected_item='stone_axe', server_axe_logs=0,
        server_inventory={'stone_axe':1, 'oak_log':108})
    result.update(changes)
    return result


class ToolHazardRecoveryTests(unittest.TestCase):
    def bundle(self):
        return dict(world_id='w', generation='pursued', observation={'inventory':{'stone_axe':1,'oak_log':108},'life_stats':{'life':10,'food':4}},
            controller_state={'tool_mission':dict(schema=1, world_id='w', stages={
                'axe_wood':dict(status='active', baseline=52, target=64, retained_gain=56,
                    verified_tool_harvests=49, last_halt_reason='tool_collection_pursuit_guard')})})

    def environment(self):
        raw=Mock();gate=StopGate()
        wrapped=ToolCollectionEnvironment(raw,self.bundle(),Mock(),'evade_hostile',gate)
        wrapped.initial_health=10.;wrapped.minimum_health=10.
        return wrapped,raw,gate

    def test_recovery_requires_recorded_pursuit_and_cannot_repeat_spent_attempt(self):
        b=self.bundle();before=copy.deepcopy(b);ledger=make_ledger(b,'evade_hostile')
        self.assertEqual(b,before)
        self.assertEqual(ledger['stages']['axe_wood'],before['controller_state']['tool_mission']['stages']['axe_wood'])
        with self.assertRaisesRegex(ValueError,'requires recovery'):make_ledger(b,'axe_wood')
        with self.assertRaises(ValueError):validate_condition_phase('evade_hostile',None)
        validate_condition_phase('evade_hostile','condition.json')
        b['controller_state']['tool_mission']=ledger
        ledger['stages']['evade_hostile']['episodes']=1
        with self.assertRaisesRegex(ValueError,'budget'):make_ledger(b,'evade_hostile')
        ledger['stages']['evade_hostile']['status']='completed'
        renewed=make_ledger(b,'evade_hostile')
        self.assertEqual(len(renewed['hazard_recovery_history']),1)
        b=self.bundle();b['controller_state']['tool_mission']['stages']['axe_wood']['last_halt_reason']=None
        with self.assertRaises(ValueError):make_ledger(b,'evade_hostile')

    def test_pursuer_is_not_safe_when_occluded_or_when_telemetry_is_missing(self):
        stability=PursuitStability()
        stability.observe(server(1,pursuit_count=1,pursuit_distance=31.9))
        for tick in range(2,102):
            stability.observe(server(tick));self.assertFalse(stability.ready(server(tick)))
        stability.observe(server(102));self.assertTrue(stability.ready(server(102)))
        self.assertFalse(stability.ready(server(103,pursuit_count=1,pursuit_distance=20)))
        missing=server(104);missing.pop('pursuit_count')
        with self.assertRaises(ValueError):stability.observe(missing)

    def test_damage_time_gap_and_airborne_state_do_not_fake_stability(self):
        for reason in ('damage','gap','airborne'):
            stability=PursuitStability()
            for tick in range(1,101):stability.observe(server(tick))
            changed=server(101)
            if reason=='damage':changed['server_health']=2.066666
            if reason=='gap':changed['server_tick']=150
            if reason=='airborne':changed.update(server_on_ground=0,server_fall_distance=2)
            stability.observe(changed);self.assertFalse(stability.ready(changed))
        with self.assertRaises(ValueError):stability.observe(server(100))

    def test_neural_evasion_preserves_root_goal_and_counts_actual_axe_harvests(self):
        wrapped,raw,gate=self.environment()
        g=server(1,server_health=10,pursuit_count=1,pursuit_distance=31.9,server_axe_logs=1)
        raw.step.return_value=({'life_stats':{'food':4}},0,False,{'workshop':g})
        action={'forward':1,'attack':1};wrapped.step(action)
        raw.step.assert_called_once_with(action);self.assertFalse(gate.requested.is_set())
        self.assertEqual(wrapped.ledger['stages']['axe_wood']['verified_tool_harvests'],50)
        self.assertEqual(wrapped.ledger['stages']['axe_wood']['baseline'],52)
        for tick in range(2,103):
            g.update(server(tick,server_health=10,server_axe_logs=1));wrapped.step({'forward':1})
        self.assertTrue(wrapped.completed)
        self.assertEqual(wrapped.ledger['stages']['axe_wood']['status'],'active')
        with self.assertRaises(RuntimeError):wrapped.step({'forward':1})
        wrapped.update_counter(server(103,server_axe_logs=2))
        self.assertEqual(wrapped.ledger['stages']['axe_wood']['verified_tool_harvests'],51)

    def test_evasion_keeps_health_food_and_tool_guards(self):
        for reason in ('health','food','tool'):
            wrapped,raw,gate=self.environment();g=server(1,server_health=10,pursuit_count=1,pursuit_distance=20)
            if reason=='health':g['server_health']=1.166666
            if reason=='tool':g['server_inventory']={'oak_log':108}
            raw.step.return_value=({'life_stats':{'food':0 if reason=='food' else 4}},0,False,{'workshop':g})
            wrapped.step({'forward':1});self.assertTrue(gate.requested.is_set());self.assertFalse(wrapped.completed)

    def test_root_goal_completion_during_recovery_stops_without_claiming_stability(self):
        wrapped,raw,gate=self.environment()
        g=server(1,pursuit_count=1,pursuit_distance=20,server_axe_logs=15,
            server_inventory={'stone_axe':1,'oak_log':116})
        raw.step.return_value=({'life_stats':{'food':4}},0,False,{'workshop':g})
        wrapped.step({'attack':1})
        self.assertTrue(gate.requested.is_set());self.assertFalse(wrapped.completed)
        self.assertEqual(wrapped.ledger['stages']['axe_wood']['status'],'completed')

    def test_cleanup_does_not_complete_recovery_when_new_food_or_damage_is_unsafe(self):
        wrapped,raw,gate=self.environment()
        for tick in range(1,102):wrapped.stability.observe(server(tick,server_health=10))
        wrapped.minimum_food=4
        self.assertTrue(wrapped.requirements(server(102,server_health=10)))
        self.assertFalse(wrapped.requirements(server(102,server_health=10,server_food=0)))
        self.assertFalse(wrapped.requirements(server(102,server_health=1.9)))

    def test_cleanup_keeps_actual_root_completion_but_never_credits_dead_player(self):
        for alive,status in ((1,'completed'),(0,'active')):
            wrapped,raw,gate=self.environment()
            wrapped.update_counter(server(1,server_alive=alive,server_axe_logs=15,
                server_inventory={'stone_axe':1,'oak_log':116}))
            self.assertEqual(wrapped.ledger['stages']['axe_wood']['status'],status)
            self.assertFalse(wrapped.completed)
            wrapped.update_counter(server(2,server_alive=0,server_axe_logs=15,server_inventory={}))
            self.assertEqual(wrapped.ledger['stages']['axe_wood']['status'],'active')
            self.assertEqual(wrapped.ledger['stages']['axe_wood']['retained_gain'],-52)

    def test_low_health_collection_profile_never_authorizes_hostile_evasion(self):
        self.assertEqual(health_limits_for_phase('axe_wood','fed_low_health')['minimum_health'],2)
        self.assertEqual(health_limits_for_phase('evade_hostile','standard')['minimum_health'],8)
        with self.assertRaisesRegex(ValueError,'low-health'):
            health_limits_for_phase('evade_hostile','fed_low_health')
        for life in (0,2.166666,8,float('nan')):
            b=self.bundle();b['observation']['life_stats']['life']=life
            with self.assertRaisesRegex(ValueError,'health'):make_ledger(b,'evade_hostile')
        b=self.bundle();b['observation']['life_stats']['life']=8.166666
        self.assertEqual(make_ledger(b,'evade_hostile')['stages']['evade_hostile']['episodes'],0)


if __name__=='__main__':unittest.main()
