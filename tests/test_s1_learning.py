import copy
import unittest
from minecraftrl.goal_queue import GoalQueue
from minecraftrl.s1_learning import phase_features, validate_data, SCOPE
from test_goal_queue import goal
try:
    import torch
except ImportError:
    torch=None


class PhaseContextTests(unittest.TestCase):
    def test_context_distinguishes_intention_result_memory_and_quantity(self):
        q=GoalQueue(dict(schema='goal_queue_v1',goals=[goal('wood',count=2)]),dict(wood=['oak_log']))
        q.observe(dict(server_available=1,error=0,server_tick=0,server_inventory={}))
        q.select();q.observe(dict(server_available=1,error=0,server_tick=1,server_inventory={'oak_log':1}))
        self.assertEqual(phase_features(q,'mine'),[0.,0.,1.,0.,.5,0.])
        self.assertEqual(phase_features(q,'collect',True),[0.,0.,0.,1.,.5,1.])

    def test_blocked_goal_has_no_policy_context(self):
        q=GoalQueue(dict(schema='goal_queue_v1',goals=[goal('stone',skill='unknown')]),{})
        q.observe(dict(server_available=1,error=0,server_tick=0,server_inventory={}))
        q.select()
        with self.assertRaises(ValueError):phase_features(q,'search')


@unittest.skipIf(torch is None,'isolated tensor runtime required')
class LearningContractTests(unittest.TestCase):
    def data(self):
        from minecraftrl.rl_policy import state_fingerprint
        heads=dict(actor={'weight':torch.zeros(2,3)},value={'weight':torch.zeros(1,3)})
        data=dict(scope=SCOPE,split='train',eligible=True,status='target_reached',
            rows=[dict(step=1,phase='collect',phase_context=[0,0,0,1,0,1],old_logp=-.5,value=0.,reward=1.,terminal=True)],
            bootstrap=0.,terminal=True,recurrent_commit='dispatch_only',initial_heads=heads,
            initial_heads_sha256=state_fingerprint(heads))
        summary=dict(execution_valid=True,worker_exit_code=0,status='target_reached',steps=1)
        return data,summary

    def test_verified_training_only_and_terminal_bootstrap(self):
        d,s=self.data();validate_data(d,s)
        for changed in (dict(split='evaluation'),dict(eligible=False),dict(status='cancelled'),dict(bootstrap=1.)):
            with self.assertRaises(ValueError):validate_data(dict(d,**changed),s)
        with self.assertRaises(ValueError):validate_data(d,dict(s,worker_exit_code=-15))

    def test_context_extension_preserves_initial_actor_and_gets_gradient(self):
        from minecraftrl.rl_policy import extend_context_head
        head=torch.nn.Sequential(torch.nn.Linear(3,4));x=torch.randn(2,1,3)
        old=head(x).detach().clone();extend_context_head(head,3,6)
        extra=torch.tensor([0.,0.,1.,0.,0.,0.]).view(1,1,6).expand(2,1,6)
        actual=head(torch.cat([x,extra],-1))
        torch.testing.assert_close(old,actual)
        actual.sum().backward()
        self.assertGreater(head[0].weight.grad[:,3:].abs().sum().item(),0)

    def test_terminal_and_truncation_returns_are_distinct(self):
        from minecraftrl.rl_core import gae
        _,terminal=gae([0.,1.],[0.,0.],99.,terminal=True,gamma=.999,lam=1.)
        _,truncated=gae([0.,0.],[0.,0.],2.,terminal=False,gamma=.999,lam=1.)
        self.assertAlmostEqual(terminal[0],.999)
        self.assertAlmostEqual(truncated[0],2*.999**2)


if __name__=='__main__':unittest.main()
