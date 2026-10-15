import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from test_rm_planner import setup_packet
from minecraftrl.rm_planner import RMPlanScheduler, fixture_plan, planning_packet


class RMPriorityTests(unittest.TestCase):
    def test_timing_context_is_in_real_planner_packet_and_survives_renamed_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            guard,scheduler,packet,current=setup_packet(folder)
            self.assertIn('scheduling',packet,'LLM does not receive goal age and service history')
            plan=fixture_plan(packet); scheduler.install(plan,packet,current)
            later=deepcopy(current); later['elapsed_s']+=40
            scheduler.update_timing(later)
            oldkey=scheduler.state['plans'][-1]['timing_ids']['search']
            oldage=scheduler.ledger.records[oldkey]['age_s']
            fresh=deepcopy(packet); fresh['schedule_revision']=scheduler.state['revision']
            plan=fixture_plan(fresh); plan['steps'][0]['id']='renamed'
            plan['steps'][-1]['depends_on']=['renamed']
            # Use a new bound packet for current facts rather than reusing an expired request.
            guard2,s2,p2,c2=setup_packet(str(Path(folder)/'second'))
            p2['schedule_revision']=scheduler.state['revision']
            p2['runtime_binding']['elapsed_s']=later['elapsed_s']
            from minecraftrl.replanning import fingerprint
            p2['snapshot_sha256']=fingerprint(p2['runtime_binding'])
            p2['remaining_control_s']=180.-later['elapsed_s']
            plan.update({k:p2[k] for k in ('request_id','evidence_id','root_id','schedule_revision')})
            scheduler.install(plan,p2,later)
            newkey=scheduler.state['plans'][-1]['timing_ids']['renamed']
            self.assertEqual(newkey,oldkey)
            self.assertEqual(scheduler.ledger.records[newkey]['age_s'],oldage)
            self.assertGreaterEqual(scheduler.ledger.records[scheduler.root_timing_id]['age_s'],60)

    def test_parent_priority_inheritance_is_used_in_actual_node_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            guard,scheduler,packet,current=setup_packet(folder)
            self.assertIn('scheduling',packet)
            plan=fixture_plan(packet); plan['steps'][0]['priority']=0; plan['steps'][-1]['priority']=100
            scheduler.install(plan,packet,current)
            result=scheduler.select(current,guard.history.snapshot())
            self.assertEqual(result['node']['id'],'search')
            self.assertGreaterEqual(result['scheduling']['effective_priority'],100)
            self.assertEqual(result['scheduling']['inherited_from'],scheduler.root_timing_id)
            restored=RMPlanScheduler(Path(folder)/'schedule.json',current)
            self.assertEqual(restored.ledger.records,scheduler.ledger.records)
            self.assertIsNone(restored.select(current,guard.history.snapshot())['node'])
