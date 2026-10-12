import copy
import unittest
from types import SimpleNamespace
import torch
from minecraftrl.temporal_bc import forward_window,windows,episode_rows
from minecraftrl.policy_tail import load_tail,tail_state,temporal_state,TEMPORAL_SCOPE


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x,first,state):
        state=state*(~first).unsqueeze(-1)+x*self.weight
        return state,state


class TemporalTests(unittest.TestCase):
    def rows(self,n=5):
        return [dict(temporal_input=torch.full((2,1,3),float(i+1),requires_grad=True),
                     features=torch.zeros(2,1,9)) for i in range(n)]

    def test_burnin_uses_updated_weights_and_window_matches_full_sequence(self):
        block=Block();tail=torch.nn.Identity();rows=self.rows();initial=torch.zeros(2,1,3)
        for weight in (1.,2.):
            with torch.no_grad():block.weight.fill_(weight)
            full=forward_window(block,tail,rows,initial,0,5)
            truncated=forward_window(block,tail,rows,initial,3,5)
            self.assertTrue(torch.equal(full[:,3:],truncated))
        self.assertEqual(float(truncated[0,-1,0]),30.)

    def test_gradient_flows_within_window_but_not_burnin_or_future(self):
        rows=self.rows();block=Block()
        out=forward_window(block,torch.nn.Identity(),rows,torch.zeros(2,1,3),2,4)
        out[:,-1,:3].sum().backward()
        for i in (0,1,4):self.assertIsNone(rows[i]['temporal_input'].grad)
        for i in (2,3):self.assertGreater(float(rows[i]['temporal_input'].grad.sum()),0.)
        self.assertGreater(float(block.weight.grad),0.)

    def test_future_change_does_not_change_past(self):
        rows=self.rows();block=Block();initial=torch.zeros(2,1,3)
        before=forward_window(block,torch.nn.Identity(),rows,initial,0,5)
        rows[-1]['temporal_input']=torch.ones(2,1,3)*100
        after=forward_window(block,torch.nn.Identity(),rows,initial,0,5)
        self.assertTrue(torch.equal(before[:,:-1],after[:,:-1]))

    def test_windows_keep_all_frames_and_partial_end(self):
        self.assertEqual(windows(18),[(0,8),(8,16),(16,18)])
        for n in (0,-1,True):
            with self.assertRaises(ValueError):windows(n)

    def net(self):
        return SimpleNamespace(lastlayer=torch.nn.Linear(3,3),final_ln=torch.nn.LayerNorm(3),
                               recurrent_layer=SimpleNamespace(blocks=torch.nn.ModuleList([Block(),Block()])))

    def checkpoint(self,net):
        return dict(scope=TEMPORAL_SCOPE,backbone_tail=tail_state(net),backbone_temporal=temporal_state(net))

    def test_temporal_checkpoint_restores_and_legacy_scope_rejects_silent_drop(self):
        net=self.net();saved=self.checkpoint(net)
        with torch.no_grad():net.recurrent_layer.blocks[-1].weight.fill_(2.)
        self.assertTrue(load_tail(net,saved))
        self.assertEqual(float(net.recurrent_layer.blocks[-1].weight),1.)
        saved['scope']='head_and_tail'
        with self.assertRaises(ValueError):load_tail(net,saved)

    def test_invalid_temporal_checkpoint_cannot_partially_mutate_tail(self):
        net=self.net();saved=self.checkpoint(net);saved['backbone_tail']['lastlayer']['weight'].zero_()
        before=tail_state(net)
        for mutation in ('missing','index','count','nan','dtype'):
            trial=copy.deepcopy(saved)
            if mutation=='missing':trial.pop('backbone_temporal')
            elif mutation in ('index','count'):trial['backbone_temporal'][mutation]=8
            elif mutation=='nan':trial['backbone_temporal']['state']['weight'].fill_(float('nan'))
            else:trial['backbone_temporal']['state']['weight']=torch.tensor(1.,dtype=torch.float64)
            with self.assertRaises(ValueError):load_tail(net,trial)
            self.assertTrue(torch.equal(before['lastlayer']['weight'],net.lastlayer.weight))

    def test_sequences_reject_gaps_split_leakage_and_nonfinite_cache(self):
        sessions=[dict(id=k,split=s,seed=i,context_start=0,start=0,end=2)
                  for i,(k,s) in enumerate((('a','train'),('b','development')))]
        plan=dict(goal='collect wood',frame_offset=0,rm_context='recorded',data_source='local_teacher',sessions=sessions)
        rows=[dict(row,session=s['id'],split=s['split'],frame=i) for s in sessions for i,row in enumerate(self.rows(2))]
        data=dict(plan=plan,rows=rows);self.assertEqual(len(episode_rows(data)),2)
        for mutation in ('gap','split','seed','nan'):
            trial=copy.deepcopy(data)
            if mutation=='gap':trial['rows'][1]['frame']=3
            elif mutation=='split':trial['rows'][1]['split']='development'
            elif mutation=='seed':trial['plan']['sessions'][1]['seed']=0
            else:trial['rows'][1]['features'][0,0,0]=float('nan')
            with self.assertRaises(ValueError):episode_rows(trial)

    def test_finetuned_backbone_is_rejected_by_head_only_ppo(self):
        from minecraftrl.rl_core import validate_training_data
        for key in ('backbone_tail','backbone_temporal'):
            with self.assertRaisesRegex(ValueError,'finetuned backbone'):
                validate_training_data(dict(initial_heads={key:{}}))
