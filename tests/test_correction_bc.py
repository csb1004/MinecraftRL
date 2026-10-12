import copy
import unittest
import torch
from minecraftrl.bc_core import validate_plan
from minecraftrl.correction_bc import correction_owner,supervised_batch,supervised_windows
from minecraftrl.temporal_bc import episode_rows,forward_window


def plan():
    sessions=[]
    for split,seed in [('train',104743),('development',104761)]:
        frames=[dict(path=f'frames/{i:06d}.png',sha256='a'*64,loss_eligible=i>=8) for i in range(10)]
        sessions.append(dict(id=split,split=split,seed=seed,context_start=0,start=0,end=10,
            eligible_for_future_masked_bc=True,expert_labels=2,frames=frames,frame_sha256=['a'*64]*10,
            handoffs=[dict(step=8,verified=True,held_key_count=0,teacher_started=True,trigger=dict(step=7,event='stalled'))]))
    return dict(schema='student_correction_v1',data_source='student_correction',masked_loss_required=True,
        goal='collect wood',frame_offset=0,rm_context='recorded',parent='parent.pt',checkpoint='parent.pt',
        reward_mode='wood-potential-v4',sessions=sessions)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x,first,state):
        state=state*(~first).unsqueeze(-1)+x*self.weight
        return state,state


class CorrectionBCTests(unittest.TestCase):
    def data(self):
        p=plan();rows=[]
        for s in p['sessions']:
            for i in range(10):
                rows.append(dict(session=s['id'],split=s['split'],frame=i,owner=correction_owner(s,i),
                    loss_eligible=i>=8,action={'buttons':torch.tensor([[[i]]])} if i>=8 else None,
                    features=torch.zeros(2,1,9),temporal_input=torch.full((2,1,3),float(i+1),requires_grad=True)))
        return dict(bc_schema=2,plan=p,rows=rows)

    def test_context_is_kept_but_has_no_action_target(self):
        data=self.data();groups=episode_rows(data)
        self.assertEqual(len(groups['train']),10)
        self.assertEqual(supervised_windows(groups['train']),[(8,10)])
        self.assertEqual(groups['train'][7]['owner'],'handoff_neutral')
        self.assertIsNone(groups['train'][7]['action'])

    def test_masks_split_and_neutral_receipt_are_strict(self):
        for mutation in ('prefix_label','teacher_mask','seed','neutral','receipt','start','count'):
            p=plan();s=p['sessions'][0]
            if mutation=='prefix_label':s['frames'][3]['loss_eligible']=True
            elif mutation=='teacher_mask':s['frames'][8]['loss_eligible']=False
            elif mutation=='seed':s['seed']=104761
            elif mutation=='neutral':s['handoffs'][0]['held_key_count']=1
            elif mutation=='receipt':s['handoffs'][0]['trigger']['step']=5
            elif mutation=='start':s['start']=1
            else:s['expert_labels']=3
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_plan(p)

    def test_cache_cannot_inject_student_labels_or_drop_context(self):
        for mutation in ('target','mask','owner','gap','schema'):
            d=self.data()
            if mutation=='target':d['rows'][0]['action']={'buttons':torch.zeros(1,1,1)}
            elif mutation=='mask':d['rows'][0]['loss_eligible']=True
            elif mutation=='owner':d['rows'][8]['owner']='student'
            elif mutation=='gap':d['rows'].pop(2)
            else:d['bc_schema']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):episode_rows(d)

    def test_only_expert_outputs_receive_direct_loss_gradient(self):
        rows=episode_rows(self.data())['train'];features=torch.ones(2,10,9,requires_grad=True)
        selected,base,targets=supervised_batch(rows,features)
        self.assertEqual(targets['buttons'].flatten().tolist(),[8,9])
        selected.sum().backward()
        self.assertEqual(float(features.grad[:,:8].abs().sum()),0.)
        self.assertGreater(float(features.grad[:,8:].abs().sum()),0.)

    def test_context_only_window_is_skipped_and_cannot_produce_loss(self):
        rows=episode_rows(self.data())['train'][:8]
        self.assertEqual(supervised_windows(rows),[])
        with self.assertRaises(ValueError):supervised_batch(rows,torch.zeros(2,8,9))

    def test_context_affects_teacher_output_and_gets_indirect_temporal_gradient(self):
        rows=episode_rows(self.data())['train'];block=Block();tail=torch.nn.Identity();initial=torch.zeros(2,1,3)
        full=forward_window(block,tail,rows,initial,2,10)
        selected,_,_=supervised_batch(rows[2:10],full)
        selected[-1,-1,:3].sum().backward()
        self.assertIsNone(rows[0]['temporal_input'].grad)
        self.assertGreater(float(rows[7]['temporal_input'].grad.abs().sum()),0.)
        altered=copy.copy(rows);altered[0]=dict(rows[0],temporal_input=rows[0]['temporal_input'].detach()+100)
        after=forward_window(block,tail,altered,initial,2,10)
        self.assertFalse(torch.equal(full[:,6:],after[:,6:]))


if __name__=='__main__':unittest.main()
