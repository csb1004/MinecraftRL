import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from minecraftrl.policy_memory import PolicyMemory,encode_state
from minecraftrl.world_store import WorldStore,digest
from minecraftrl.rl_policy import state_fingerprint


def policy():
    return SimpleNamespace(torch=torch,device='cpu',goal_embed=np.zeros((1,512),np.float32),
        condition_metadata={'embedding_sha256':'a'*64,'cond_scale':7.},
        agent=SimpleNamespace(cond_scale=7.,hidden_state=[(torch.ones((2,1,3),dtype=torch.bool),
            (torch.arange(12,dtype=torch.float32).reshape(2,3,2),torch.ones((2,3,2),dtype=torch.float16)))]))


class PolicyMemoryTests(unittest.TestCase):
    task=dict(task_id='wood',revision=1,goal='wood')
    identity={'fixture':'fixed-weights'}

    def saved(self,root):
        p=policy();template=encode_state(p.agent.hidden_state)[0]
        memory=PolicyMemory(p,self.identity,template)
        self.assertFalse(memory.on_reset(self.task,'world-a')['restored'])
        memory.inferred();payload=memory.snapshot(root)
        bundle=dict(archive_sha256='b'*64,policy_memory_path=payload['path'],
            policy_memory=dict(payload['metadata'],file_sha256=digest(payload['path']),world_archive_sha256='b'*64))
        return p,template,payload,bundle

    def test_roundtrip_preserves_recurrent_values_rng_and_consumes_source_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,template,_,bundle=self.saved(Path(tmp));expected_state=state_fingerprint(p.agent.hidden_state)
            expected_random=torch.rand(20)
            p.agent.hidden_state=[];torch.manual_seed(999)
            memory=PolicyMemory(p,self.identity,template,bundle)
            receipt=memory.on_reset(self.task,'world-a')
            self.assertTrue(receipt['restored']);self.assertEqual(state_fingerprint(p.agent.hidden_state),expected_state)
            torch.testing.assert_close(torch.rand(20),expected_random,rtol=0,atol=0)
            p.agent.hidden_state=[]
            self.assertFalse(memory.on_reset(self.task,'world-a')['restored'])
            self.assertEqual(p.agent.hidden_state,[])

    def test_changed_world_revision_goal_or_condition_does_not_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,template,_,bundle=self.saved(Path(tmp))
            for task,world,condition in [(dict(self.task,revision=2),'world-a',{}),
                    (dict(self.task,goal='dirt'),'world-a',{}),(self.task,'world-b',{}),
                    (self.task,'world-a',{'cond_scale':2.}),(self.task,'world-a',{'embedding_sha256':'c'*64})]:
                p=policy();p.condition_metadata.update(condition);p.agent.hidden_state=[]
                m=PolicyMemory(p,self.identity,template,bundle)
                self.assertEqual(m.on_reset(task,world)['reason'],'task_or_condition_changed')
                self.assertEqual(p.agent.hidden_state,[])

    def test_tampered_model_archive_tensor_and_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,template,payload,bundle=self.saved(Path(tmp))
            for mutation in ('identity','archive','fingerprint','shape'):
                bad=copy.deepcopy(bundle)
                if mutation=='identity':bad['policy_memory']['identity']={}
                if mutation=='archive':bad['archive_sha256']='c'*64
                if mutation=='fingerprint':bad['policy_memory']['memory_sha256']='0'*64
                if mutation=='shape':bad['policy_memory']['tree']={}
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    PolicyMemory(p,self.identity,template,bad)
            Path(payload['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):PolicyMemory(p,self.identity,template,bundle)
            with self.assertRaises(ValueError):encode_state([torch.tensor([float('nan')])])

    def test_unstarted_session_preserves_existing_memory_but_empty_new_session_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,template,payload,bundle=self.saved(root)
            m=PolicyMemory(p,self.identity,template,bundle)
            self.assertEqual(m.snapshot(root/'unused')['path'],payload['path'])
            empty=PolicyMemory(p,self.identity,template)
            self.assertIsNone(empty.snapshot(root/'unused'))
            empty.on_reset(self.task,'world-a')
            self.assertIsNone(empty.snapshot(root/'unused'))

    def test_world_generation_atomically_binds_policy_memory(self):
        from test_world_store import WorldStoreTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,template,payload,_=self.saved(root)
            world,task,obs=WorldStoreTests().fixture(root)
            # Exercise the maintained multi-goal world-store contract.
            from minecraftrl.scheduled_tasks import ScheduledTasks
            state=ScheduledTasks();state.apply(dict(command_id='a',task_id='wood',op='request',goal='wood',count=64),{})
            task=state.checkpoint('world-a')
            with WorldStore(root/'store','source') as store:
                generation=store.commit(world,task,obs,policy_memory=payload)
                loaded=store.load();self.assertEqual(loaded['policy_memory']['world_archive_sha256'],loaded['archive_sha256'])
                self.assertEqual(digest(loaded['policy_memory_path']),digest(payload['path']))
                with patch('minecraftrl.world_store.atomic_json',side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):store.commit(world,task,obs,policy_memory=payload)
                self.assertEqual(store.load()['generation'],generation.name)
                (generation/'policy-memory.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):store.load()

    def test_cli_memory_rejects_unsupported_paths(self):
        from minecraftrl.cli import parser,validate_policy_memory_args
        good=['session','--policy','steve1','--task-session','--world-slot','test','--steve-prompt-profile','official-visual','--task-policy-memory']
        validate_policy_memory_args(parser().parse_args(good))
        for flag in ('--rl-collect','--craft-probe','--craft-recipe-book','--craft-workshop'):
            with self.assertRaises(ValueError):validate_policy_memory_args(parser().parse_args(good+[flag]))
        for argv in (['episode','--policy','vpt','--task-policy-memory'],
                     ['session','--policy','steve1','--task-session','--world-slot','test','--task-policy-memory']):
            with self.assertRaises(ValueError):validate_policy_memory_args(parser().parse_args(argv))

    def test_stop_discards_action_and_saves_last_inference_only_after_release_and_quit(self):
        from minecraftrl import task_runtime
        from minecraftrl.runtime import StopGate
        from minecraftrl.tasks import CommandBus
        from test_task_session import FakeEnvironment
        from test_world_store import WorldStoreTests
        gate=StopGate();bus=CommandBus();order=[];p=policy()
        bus.submit(dict(command_id='a',op='request',task_id='wood',goal='wood',count=64))
        p.reset=lambda goal:None
        def act(frame):
            order.append('infer');p.agent.hidden_state[0][1][0].add_(1)
            gate.stop('during_inference');return 'oak_log'
        p.act=act;p.close=lambda:order.append('policy_close')
        memory=PolicyMemory(p,self.identity,encode_state(p.agent.hidden_state)[0])
        capture=memory.snapshot
        def snapshot(output):
            import json
            self.assertTrue(gate.requested.is_set())
            events=[json.loads(line) for line in (Path(output)/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='stop_acknowledged' for e in events))
            self.assertEqual(order[-2:],['release','save'])
            order.append('snapshot');return capture(output)
        memory.snapshot=snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);world,_,obs=WorldStoreTests().fixture(root)
            class Environment(FakeEnvironment):
                def observation(self):return dict(super().observation(),location_stats=obs['location_stats'])
                def release(self):order.append('release');return super().release()
                def save_and_quit(self):order.append('save');return world
            env=Environment()
            with WorldStore(root/'store','test') as store:
                report=task_runtime.run_task_session(p,env,root/'output',seed=1,limit_s=1,
                    gate=gate,bus=bus,save_screen=lambda *a:None,world_store=store,policy_memory=memory)
                self.assertEqual(report['world_save']['status'],'saved')
                self.assertNotIn('policy_memory_error',report)
                saved=store.load()['policy_memory']
                self.assertEqual(saved['inferred_frames'],1)
                self.assertEqual(saved['memory_sha256'],state_fingerprint(p.agent.hidden_state))
            self.assertEqual(env.actions,[])
            self.assertEqual(order.count('infer'),1)
            self.assertLess(order.index('snapshot'),order.index('policy_close'))


if __name__=='__main__':unittest.main()
