import unittest
from minecraftrl.rl_core import episode_returns, tick_delta


class RLBatchTests(unittest.TestCase):
    def test_time_limit_training_requires_committed_memory_bootstrap(self):
        from minecraftrl.rl_core import validate_training_data
        data=dict(eligible=True,split='train',scope='head_only_frozen_features',status='time_limit',
                  rows=[dict(old_logp=-1.,value=0.,reward=0.)],condition={})
        with self.assertRaises(ValueError):validate_training_data(data)
        data['condition']['recurrent_commit_mode']='dispatch_commit'
        validate_training_data(data)

    def test_uncommitted_action_does_not_advance_recurrent_state(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import MagicMock,patch
        from minecraftrl.rl_policy import RLPolicy
        policy=object.__new__(RLPolicy);policy.first=False;policy.rows=[];policy.pending=None
        policy.reward_mode='sparse'
        policy.torch=SimpleNamespace(no_grad=nullcontext,isfinite=lambda v:SimpleNamespace(all=lambda:True))
        head=SimpleNamespace(sample=lambda *a,**k:{'buttons':MagicMock()},logprob=lambda *a:SimpleNamespace(item=lambda:-1.))
        policy.agent=SimpleNamespace(hidden_state='committed',policy=SimpleNamespace(pi_head=head),_agent_action_to_env=lambda a:{'attack':1})
        observed=[]
        def features(screen):observed.append(policy.agent.hidden_state);return MagicMock(),'pending'
        policy.features=features;policy.value=lambda f:SimpleNamespace(item=lambda:2.)
        with patch('minecraftrl.rl_policy.guided',return_value={'buttons':0}):action=policy.act(None)
        self.assertEqual(policy.agent.hidden_state,'committed')
        self.assertEqual(policy.bootstrap(None),2.);self.assertEqual(observed,['committed','committed'])
        policy.commit(action,0.,False)
        self.assertEqual(policy.agent.hidden_state,'pending');self.assertEqual(len(policy.rows),1)

    def test_startup_failure_keeps_original_error_without_policy_reset(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from minecraftrl.rl_collect import run_rl
        closed=[];policy=SimpleNamespace(torch=None,dispose=lambda:closed.append(True))
        report=dict(status='error',error='renderer unavailable',execution_valid=False,steps=0)
        with tempfile.TemporaryDirectory() as directory,patch('minecraftrl.rl_collect.run_episode',return_value=report):
            result=run_rl(policy,None,Path(directory),split='eval',seed=7,gate=None)
        self.assertEqual(result['error'],'renderer unavailable');self.assertEqual(closed,[True])

    def test_fixed_warmup_records_global_queue_without_using_it_as_visible_pixel_gate(self):
        from types import SimpleNamespace
        from minecraftrl.rl_collect import RLEnvironment
        class Env:
            def __init__(self,ready):self.calls=0;self.ready=ready;self.raw=SimpleNamespace(action_space=SimpleNamespace(noop=lambda:{}))
            def reset(self):pass
            def step(self,action):
                self.calls+=1
                return {},0,False,{'workshop':dict(server_available=1,error=0,server_inventory={},server_tick=self.calls,render_ready=int(self.ready))}
        env=Env(True);wrapper=RLEnvironment(env,None,strict_ticks=True);wrapper.reset()
        self.assertEqual(env.calls,60)
        pending=RLEnvironment(Env(False),None,strict_ticks=True);pending.reset()
        self.assertEqual(pending.initial['server']['render_ready'],0)

    def test_initial_archive_rejects_seed_and_hash_mismatch(self):
        import tempfile,json,zipfile,gzip,struct
        from pathlib import Path
        from minecraftrl.rl_worlds import checked_world
        from minecraftrl.world_store import digest
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'initial-world.zip'
            def text(value):return struct.pack('>H',len(value))+value.encode()
            nbt=b'\x0a'+text('')+b'\x0a'+text('Data')+b'\x0a'+text('WorldGenSettings')+b'\x04'+text('seed')+struct.pack('>q',7)+b'\0\0\0'
            with zipfile.ZipFile(path,'w') as z:z.writestr('./saves/test/level.dat',gzip.compress(nbt))
            path.with_suffix('.json').write_text(json.dumps(dict(seed=7,sha256=digest(path))))
            self.assertEqual(checked_world(path,7)['seed'],7)
            with self.assertRaises(ValueError):checked_world(path,8)
            # A matching archive hash cannot make a false claimed seed valid.
            path.with_suffix('.json').write_text(json.dumps(dict(seed=8,sha256=digest(path))))
            with self.assertRaisesRegex(ValueError,'Saved world seed'):checked_world(path,8)
            path.with_suffix('.json').write_text(json.dumps(dict(seed=7,sha256=digest(path))))
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):checked_world(path,7)

    def test_episode_boundaries_do_not_leak_reward(self):
        data=[dict(rows=[dict(reward=0.,value=0.)],bootstrap=0.,terminal=True),
              dict(rows=[dict(reward=1.,value=0.)],bootstrap=0.,terminal=True)]
        advantage,returns=episode_returns(data)
        self.assertEqual(returns,[0.,1.]);self.assertEqual(advantage,returns)

    def test_tick_contract_rejects_missing_or_skipped_ticks(self):
        self.assertEqual(tick_delta({'server_tick':10},{'server_tick':11}),1)
        for before,after in (({},{}),({'server_tick':1},{'server_tick':3}),
                             ({'server_tick':1},{'server_tick':1})):
            with self.assertRaises(ValueError):tick_delta(before,after)
