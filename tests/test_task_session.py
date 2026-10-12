import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from minecraftrl.tasks import CommandBus
from minecraftrl import task_runtime


class FakeEnvironment:
    def __init__(self):
        self.inventory={'dirt':2}; self.resets=0; self.actions=[]; self.releases=0
        self.last_observation=self.observation(); self.closed=False
    def observation(self): return {'inventory':dict(self.inventory),'pov':None}
    def seed(self, seed): self.seed_value=seed
    def reset(self):
        self.resets+=1; self.last_observation=self.observation(); return self.last_observation
    def step(self, action):
        self.actions.append(action)
        self.inventory[action]=self.inventory.get(action,0)+1
        self.last_observation=self.observation()
        return self.last_observation,0,False,{}
    def release(self):
        self.releases+=1; self.last_observation=self.observation(); return 'neutral_command_sent'
    def close(self): self.closed=True


def cmd(op, cid, **kw): return dict(command_id=cid,op=op,task_id='gather',**kw)


class SessionTests(unittest.TestCase):
    def test_rm_acceptance_prevents_next_inference_and_dispatch(self):
        bus=CommandBus();env=FakeEnvironment()
        bus.submit(cmd('request','terminal',goal='dirt',count=1))
        class Policy:
            calls=0
            def reset(self,goal):pass
            def act(self,screen):self.calls+=1;return 'dirt'
            def close(self):pass
        policy=Policy()
        with tempfile.TemporaryDirectory() as folder:
            report=task_runtime.run_task_session(policy,env,folder,seed=1,limit_s=1,bus=bus,save_screen=lambda *a:None)
            events=[json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
        accepted=next(i for i,e in enumerate(events) if e['event']=='inventory_target_reached')
        self.assertFalse(any(e['event'] in ('inference_started','policy_dispatch_started') for e in events[accepted+1:]))
        self.assertEqual(policy.calls,1)
        self.assertEqual(report['task']['status'],'completed')

    def test_directory_inbox_accepts_atomic_commands_and_rejects_completion_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus=CommandBus(); events=[]
            inbox=task_runtime.DirectoryInbox(tmp,bus,lambda event,**kw:events.append((event,kw)))
            inbox.start()
            try:
                task_runtime.write_command(tmp,cmd('request','a',goal='wood',count=4))
                (Path(tmp)/'invalid.json').write_text(json.dumps(cmd('completed','b')))
                deadline=time.monotonic()+2
                while len(events)<2 and time.monotonic()<deadline: time.sleep(.005)
            finally: inbox.close()
            self.assertEqual([x['command_id'] for x in bus.drain()],['a'])
            self.assertTrue(any(x[0]=='command_rejected' for x in events))

    def test_same_world_modify_during_inference_pause_resume_and_inventory_completion(self):
        bus=CommandBus(); env=FakeEnvironment()
        bus.submit(cmd('request','a',goal='wood',count=4))
        class Policy:
            def __init__(self): self.calls=0; self.goals=[]
            def reset(self, goal): self.goals.append(goal); self.goal=goal
            def act(self, frame):
                self.calls+=1
                if self.calls==2:
                    bus.submit(cmd('modify','b',expected_revision=1,goal='dirt',count=4))
                    bus.submit(cmd('pause','c',expected_revision=2))
                return 'oak_log' if self.goal=='collect wood' else 'dirt'
            def close(self): pass
        policy=Policy()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp); stop=threading.Event(); watcher_errors=[]
            def resume_after_ack():
                deadline=time.monotonic()+3
                while not stop.wait(.005) and time.monotonic()<deadline:
                    path=output/'events.jsonl'
                    if not path.exists(): continue
                    try: rows=[json.loads(line) for line in path.read_text().splitlines()]
                    except json.JSONDecodeError: continue
                    if any(r['event']=='command_acknowledged' and r['command_id']=='c' for r in rows):
                        before=len(env.actions); time.sleep(.03)
                        if len(env.actions)!=before: watcher_errors.append('action during pause')
                        bus.submit(cmd('resume','d',expected_revision=2)); return
            thread=threading.Thread(target=resume_after_ack); thread.start()
            try:
                report=task_runtime.run_task_session(policy,env,output,seed=1,limit_s=3,bus=bus,
                                                    save_screen=lambda *args:None)
            finally: stop.set(); thread.join()
            rows=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
        self.assertEqual(watcher_errors,[])
        self.assertEqual(env.resets,1)
        self.assertEqual(env.actions,['oak_log','dirt','dirt','dirt','dirt'])
        self.assertEqual(report['task']['revision'],2)
        self.assertEqual(report['task']['baseline'],2)
        self.assertEqual(report['task']['progress'],4)
        self.assertEqual(report['status'],'task_completed')
        self.assertTrue(report['execution_valid'])
        self.assertTrue(any(r['event']=='action_discarded' for r in rows))
        self.assertTrue(env.closed)

    def test_cancel_during_inference_discards_action(self):
        bus=CommandBus(); env=FakeEnvironment()
        bus.submit(cmd('request','a',goal='dirt',count=4))
        class Policy:
            def reset(self, goal): pass
            def act(self, frame):
                bus.submit(cmd('cancel','b',expected_revision=1)); return 'dirt'
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            report=task_runtime.run_task_session(Policy(),env,tmp,seed=1,limit_s=1,bus=bus,
                                                save_screen=lambda *args:None)
        self.assertEqual(env.actions,[])
        self.assertEqual(report['task']['status'],'cancelled')
        self.assertFalse(report['execution_valid'])


if __name__=='__main__': unittest.main()
