import unittest


class SemanticEvaluationTests(unittest.TestCase):
    def test_context_bound_case_rejects_missing_or_different_input_even_with_same_prompt(self):
        from minecraftrl.semantic_evaluation import evaluate
        case={'id':'paused','input_sha256':'c'*64,'expected':{'intent':'TASK_RESUME','goal':None,'count':None}}
        record={'case_id':'paused','prompt_sha256':'a'*64,'response':{'status':'incomplete'}}
        for row in [record,dict(record,input_sha256='d'*64)]:
            with self.assertRaisesRegex(ValueError,'input_context_mismatch'):
                evaluate([case],[row],current_prompt_sha256='a'*64)
        report=evaluate([case],[dict(record,input_sha256='c'*64)],current_prompt_sha256='a'*64)
        self.assertEqual(report['response_errors'],1)

    def test_optional_topic_difference_does_not_count_as_wrong_intent(self):
        from minecraftrl.semantic_evaluation import evaluate
        case={'id':'question','expected':{'intent':'QUESTION','goal':'wood','count':None}}
        response={'status':'completed','output':[{'type':'message','content':[
            {'type':'output_text','text':'{"intent":"QUESTION","goal":null,"count":null}'}]}]}
        report=evaluate([case],[{'case_id':'question','prompt_sha256':'a'*64,'response':response}],current_prompt_sha256='a'*64)
        self.assertEqual(report['label_matches'],1)
        self.assertEqual(report['candidate_matches'],0)
        self.assertEqual(report['results'][0]['mismatched_fields'],['goal'])

    def test_historical_misclassification_is_not_repaired_or_counted_as_current_prompt(self):
        from minecraftrl.semantic_evaluation import evaluate
        case={'id':'future','expected':{'intent':'TASK_PROPOSAL','goal':'wood','count':None}}
        response={'status':'completed','output':[{'type':'message','content':[
            {'type':'output_text','text':'{"intent":"CHAT","goal":"wood","count":null}'}]}]}
        report=evaluate([case],[{'case_id':'future','prompt_sha256':'a'*64,'response':response}],current_prompt_sha256='b'*64)
        self.assertEqual(report['label_matches'],0)
        self.assertEqual(report['current_prompt_evaluated'],0)
        self.assertEqual(report['results'][0]['actual']['intent'],'CHAT')
        self.assertEqual(report['results'][0]['mismatched_fields'],['intent'])

    def test_unrun_and_invalid_response_are_distinct_from_valid_label_mismatch(self):
        from minecraftrl.semantic_evaluation import evaluate
        cases=[{'id':name,'expected':{'intent':'CHAT','goal':None,'count':None}} for name in ['bad','absent']]
        report=evaluate(cases,[{'case_id':'bad','prompt_sha256':'a'*64,'response':{'status':'incomplete'}}],current_prompt_sha256='a'*64)
        self.assertEqual((report['evaluated'],report['unrun'],report['response_errors'],report['label_mismatches']),(1,1,1,0))
        self.assertEqual(report['results'][1]['status'],'unrun')

    def test_duplicate_unknown_cases_and_missing_provenance_are_rejected(self):
        from minecraftrl.semantic_evaluation import evaluate
        case={'id':'known','expected':{'intent':'CHAT','goal':None,'count':None}}
        row={'case_id':'known','prompt_sha256':'a'*64,'response':{}}
        for cases,records in [([case,case],[]),([case],[row,row]),([case],[dict(row,case_id='unknown')]),
                              ([case],[{'case_id':'known','response':{}}])]:
            with self.assertRaises(ValueError): evaluate(cases,records,current_prompt_sha256='b'*64)

    def test_proposal_memory_remains_nonexecuting_after_eviction_and_reopen(self):
        import tempfile
        from pathlib import Path
        from minecraftrl.memory import DialogueMemory
        from minecraftrl.semantic import Conversation
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.json'; conversation=Conversation(memory=DialogueMemory(path))
            result=conversation.propose('나중에 나무를 모아보자',{'intent':'TASK_PROPOSAL','goal':'wood','count':None},None)
            self.assertIsNone(result['command'])
            for index in range(14): conversation.remember('user',str(index))
            restored=Conversation(memory=DialogueMemory(path))
            self.assertTrue(any(x['category']=='unconfirmed_proposal' for x in restored.memory.summary))
            self.assertIsNone(restored.pending)
            self.assertIsNone(restored.confirm('확인 old',None)['command'])
