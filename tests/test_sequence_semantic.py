import unittest
from minecraftrl.semantic import Conversation,validate_candidate
from minecraftrl.tasks import TaskState


class SequenceSemanticTests(unittest.TestCase):
    def test_evaluation_detects_reordered_plan(self):
        import json
        from minecraftrl.semantic_evaluation import evaluate
        expected=dict(intent='TASK_PLAN_REQUEST',goal=None,count=None,steps=[{'goal':'wood','count':2},{'goal':'dirt','count':3}])
        actual=dict(expected,steps=list(reversed(expected['steps'])))
        response={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(actual)}]}]}
        result=evaluate([{'id':'plan','expected':expected}],[dict(case_id='plan',prompt_sha256='a'*64,response=response)],current_prompt_sha256='a'*64)
        self.assertEqual(result['candidate_matches'],0)
        self.assertEqual(result['results'][0]['mismatched_fields'],['steps'])

    def test_plan_requires_confirmation_and_preserves_order(self):
        candidate={'intent':'TASK_PLAN_REQUEST','goal':None,'count':None,
                   'steps':[{'goal':'wood','count':2},{'goal':'dirt','count':3}]}
        c=Conversation(); p=c.propose('나무 둘, 다음 흙 셋',candidate,None)
        self.assertIsNone(p['command'])
        command=c.confirm('확인 '+p['confirmation_id'],None)['command']
        state=TaskState(); state.apply(command,{})
        self.assertEqual(state.task['plan'],candidate['steps'])

    def test_plan_fields_cannot_smuggle_actions_into_nonexecuting_intent(self):
        for candidate in [dict(intent='CHAT',goal=None,count=None,steps=[{'goal':'wood','count':2}]),
                          dict(intent='TASK_PLAN_REQUEST',goal='wood',count=2,steps=[{'goal':'wood','count':2}]),
                          dict(intent='TASK_PLAN_REQUEST',goal=None,count=None,steps=[])]:
            with self.assertRaises(ValueError): validate_candidate(candidate)
        self.assertEqual(validate_candidate(dict(intent='CHAT',goal=None,count=None,steps=None)),dict(intent='CHAT',goal=None,count=None))
