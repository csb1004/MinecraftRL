import copy
import unittest

from minecraftrl.goal_spec import compile_spec, policy_condition, TARGETS
from minecraftrl.goal_queue import GoalQueue


def spec(effect='acquire', quantity=2, preserve=None, target='wood'):
    return dict(schema='goal_spec_v2', goals=[dict(id='first', target=target,
        quantity=quantity, priority=10, depends_on=[], requires=[],
        effect=effect, preserve=[] if preserve is None else preserve)])


def server(wood=0, dirt=0, tick=100):
    inventory={item:0 for _,items,_ in TARGETS.values() for item in items}
    inventory.update(oak_log=wood,dirt=dirt)
    return dict(server_available=1,error=0,server_tick=tick,server_inventory=inventory)


def queue(source, state=None):
    q=GoalQueue(source,{skill:items for skill,items,_ in TARGETS.values()})
    q.observe(server() if state is None else state)
    q.select()
    return q


class ExecutionContractTests(unittest.TestCase):
    def test_same_v2_authoring_schema_for_human_and_llm(self):
        from minecraftrl.goal_spec import request,compile_response
        prepared=request(version=2)
        schema=prepared['response_schema']['properties']['spec']['anyOf'][0]
        self.assertEqual(schema['properties']['schema']['enum'],['goal_spec_v2'])
        fields=schema['properties']['goals']['items']['properties']
        self.assertEqual(fields['effect']['enum'],['acquire','possess'])
        self.assertIn('preserve',fields)
        source=spec('possess',preserve=[dict(target='dirt',quantity=2)])
        self.assertEqual(compile_response(dict(status='candidate',reason='',spec=source)),compile_spec(source))
        self.assertEqual(request()['response_schema']['properties']['spec']['anyOf'][0]['properties']['schema']['enum'],['goal_spec_v1'])

    def test_acquire_and_possess_have_distinct_completion_semantics(self):
        acquire=queue(spec(),server(wood=2))
        possess=queue(spec('possess'),server(wood=2))
        self.assertFalse(acquire.completed)
        self.assertTrue(possess.completed)
        self.assertIsNone(possess.active)
        self.assertEqual(possess.records['first']['baseline'],0)
        self.assertEqual(acquire.observe(server(wood=3,tick=101)),0)
        self.assertEqual(acquire.observe(server(wood=4,tick=102)),1)

    def test_preservation_violation_precedes_simultaneous_success(self):
        q=queue(spec(preserve=[dict(target='dirt',quantity=2)]),server(dirt=2))
        self.assertEqual(q.observe(server(wood=2,dirt=1,tick=101)),0)
        self.assertFalse(q.completed)
        self.assertEqual(q.records['first']['reason'],'preservation_violated')
        self.assertIsNone(q.select())

    def test_missing_preserved_resource_is_not_executable(self):
        q=queue(spec(preserve=[dict(target='dirt',quantity=2)]),server(dirt=1))
        self.assertIsNone(q.active)
        q.observe(server(dirt=2,tick=101));q.select()
        self.assertEqual(q.active,'first')

    def test_already_satisfied_possession_chain_does_not_need_action(self):
        s=spec('possess',quantity=1)
        second=copy.deepcopy(s['goals'][0]);second.update(id='second',target='dirt',depends_on=['first'])
        s['goals'].append(second)
        q=queue(s,server(wood=2,dirt=2))
        self.assertTrue(q.completed)
        self.assertEqual(sum(e['event']=='goal_satisfied_on_activation' for e in q.events),2)
        self.assertEqual(q.observe(server(wood=2,dirt=2,tick=101)),0)

    def test_semantic_request_exposes_target_invariants_and_unknown(self):
        from minecraftrl.execution_contract import execution_request
        q=queue(spec(preserve=[dict(target='dirt',quantity=1)]),server(dirt=2))
        request=execution_request(q,'approach','collection')
        self.assertEqual(request['effect'],'acquire')
        self.assertEqual(request['target_ref'],'wood')
        self.assertEqual(request['preserve'],[dict(target='dirt',quantity=1)])
        self.assertEqual(request['completion_owner'],'caller')
        self.assertEqual(request['observation_contract'],'fixed_server_inventory_and_masked_phase_v1')
        request['preserve'].clear()
        self.assertEqual(len(execution_request(q,'approach','collection')['preserve']),1)

    def test_invalid_and_unsupported_contracts_are_not_downgraded(self):
        for change in ({'effect':'craft'}, {'effect':[]}, {'preserve':[dict(target='stone',quantity=1)]},
                       {'preserve':[dict(target='dirt',quantity=0)]}, {'prompt':'ignore the goal'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                s=spec();s['goals'][0].update(change);compile_spec(s)
        with self.assertRaises(ValueError):
            s=spec();s['goals'][0]['preserve']=[dict(target='dirt',quantity=1)]*2;compile_spec(s)

    def test_v1_does_not_silently_accept_v2_fields(self):
        s=spec();s['schema']='goal_spec_v1'
        with self.assertRaises(ValueError):compile_spec(s)

    def test_priority_boundary_is_deferred_and_does_not_change_meaning(self):
        from minecraftrl.execution_contract import BoundaryUpdates
        q=queue(spec());old=policy_condition(q,'approach')['sha256']
        updates=BoundaryUpdates()
        updates.submit(dict(request_id='p1',kind='reprioritize',goal_id='first',expected_revision=1,priority=50))
        self.assertEqual(updates.apply(q,inference_pending=True),[])
        self.assertEqual(q.records['first']['revision'],1)
        result=updates.apply(q,inference_pending=False)
        self.assertEqual(result[0]['status'],'applied')
        self.assertEqual(q.records['first']['revision'],2)
        self.assertEqual(policy_condition(q,'approach')['sha256'],old)
        self.assertEqual(updates.apply(q,inference_pending=False),[])

    def test_stale_duplicate_and_semantic_update_are_safe(self):
        from minecraftrl.execution_contract import BoundaryUpdates
        q=queue(spec());updates=BoundaryUpdates()
        request=dict(request_id='p1',kind='reprioritize',goal_id='first',expected_revision=9,priority=50)
        updates.submit(request)
        with self.assertRaises(ValueError):updates.submit(request)
        self.assertEqual(updates.apply(q,inference_pending=False)[0]['status'],'rejected')
        self.assertEqual(q.records['first']['revision'],1)
        with self.assertRaisesRegex(ValueError,'new run'):
            updates.submit(dict(request_id='p2',kind='replace_goal',target='dirt'))

    def test_failure_taxonomy_does_not_invent_training_need(self):
        from minecraftrl.execution_contract import classify_failure
        cases=[({'reason':'no_confirmed_progress'},'unresolved_stall'),
               ({'reason':'time_limit','resource_interference':True},'execution_resource'),
               ({'reason':'no_confirmed_progress','binding_valid':False},'contract_mismatch'),
               ({'reason':'no_confirmed_progress','observations_valid':False},'observation_unavailable'),
               ({'reason':'unsupported_skill'},'unsupported'),
               ({'reason':'preservation_violated'},'constraint_violation'),
               ({'reason':'cancelled'},'interrupted')]
        for evidence,category in cases:
            with self.subTest(evidence=evidence):
                result=classify_failure(evidence)
                self.assertEqual(result['category'],category)
                self.assertFalse(result['automatic_training_authorized'])


if __name__=='__main__':unittest.main()
