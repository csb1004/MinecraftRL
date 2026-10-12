import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from minecraftrl.tasks import TaskState


class WorldStoreTests(unittest.TestCase):
    def test_save_barrier_patch_is_scoped_and_refuses_unexpected_source(self):
        from minecraftrl.world_save_patch import patch_source
        source='public void stop() {\n                          mc.mouseHelper.setHumanInput(true);\n                          mc.world.sendQuittingDisconnectingPacket();\n'
        patched=patch_source(source)
        self.assertIn('saveAllPlayerData()',patched)
        self.assertIn('save(false, true, true)',patched)
        self.assertLess(patched.index('saveAllPlayerData()'),patched.index('mc.world.sendQuittingDisconnectingPacket()'))
        with self.assertRaises(ValueError): patch_source('unknown upstream')

    def test_explicit_new_request_after_terminal_task_has_fresh_baseline(self):
        state=TaskState()
        state.apply(dict(command_id='a',task_id='g',op='request',goal='dirt',count=2),{})
        state.observe({'dirt':2},1)
        task=state.apply(dict(command_id='b',task_id='g',op='request',goal='dirt',count=4),{'dirt':2})
        self.assertEqual(task['revision'],2)
        self.assertEqual(task['baseline'],2)
        self.assertEqual(task['progress'],0)

    def test_paused_progress_tracks_real_inventory_without_auto_completion(self):
        state=TaskState()
        state.apply(dict(command_id='a',task_id='g',op='request',goal='dirt',count=2),{})
        state.apply(dict(command_id='b',task_id='g',op='pause',expected_revision=1),{})
        state.observe({'dirt':2},1)
        self.assertEqual(state.task['progress'],2)
        self.assertEqual(state.task['status'],'paused')

    def fixture(self, root):
        world=root/'live'; world.mkdir(); (world/'level.dat').write_bytes(b'level')
        (world/'region').mkdir(); (world/'region/r.0.0.mca').write_bytes(b'changed blocks')
        state=TaskState(); state.apply(dict(command_id='a',task_id='g',op='request',goal='dirt',count=4),{})
        state.observe({'dirt':2},1)
        observation={'inventory':{'dirt':2},'location_stats':{'xpos':1.,'ypos':64.,'zpos':2.}}
        return world,state.checkpoint('world-a'),observation

    def test_roundtrip_preserves_world_and_task_and_rejects_bad_observation(self):
        from minecraftrl.world_store import WorldStore, validate_restored_observation
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); world,task,obs=self.fixture(root)
            with WorldStore(root/'store','companion') as store:
                self.assertIsNone(store.load())
                store.commit(world,task,obs)
            with WorldStore(root/'store','companion') as store:
                bundle=store.load(); self.assertEqual(bundle['task_checkpoint'],task)
                with zipfile.ZipFile(bundle['archive']) as archive:
                    self.assertEqual(archive.read('./saves/live/region/r.0.0.mca'),b'changed blocks')
                validate_restored_observation(bundle,obs)
                with self.assertRaises(ValueError): validate_restored_observation(bundle,dict(obs,inventory={}))
                bad=dict(obs,location_stats=dict(obs['location_stats'],xpos=100))
                with self.assertRaises(ValueError): validate_restored_observation(bundle,bad)

    def test_corrupt_save_is_rejected_and_previous_generation_retained(self):
        from minecraftrl.world_store import WorldStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); world,task,obs=self.fixture(root)
            with WorldStore(root/'store','companion') as store:
                first=store.commit(world,task,obs)
                second=store.commit(world,task,obs)
                self.assertTrue((first/'world.zip').exists())
                (second/'world.zip').write_bytes(b'corrupt')
                with self.assertRaises(ValueError): store.load()

    def test_interrupted_publication_preserves_last_complete_generation(self):
        from unittest.mock import patch
        from minecraftrl.world_store import WorldStore
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); world,task,obs=self.fixture(root)
            with WorldStore(root/'store','companion') as store:
                first=store.commit(world,task,obs)
                with patch('minecraftrl.world_store.atomic_json',side_effect=OSError('disk full')):
                    with self.assertRaises(OSError): store.commit(world,task,obs)
                self.assertEqual(store.load()['generation'],first.name)

    def test_invalid_slots_and_parallel_writers_rejected(self):
        from minecraftrl.world_store import WorldStore
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): WorldStore(tmp,'../elsewhere')
            with WorldStore(tmp,'companion'):
                with self.assertRaises((OSError,ValueError)):
                    with WorldStore(tmp,'companion'): pass

    def test_archive_paths_are_validated_before_java_loader(self):
        from minecraftrl.world_store import validate_archive
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('./saves/live/level.dat','x')
                archive.writestr('./saves/live/../../../outside','x')
            with self.assertRaises(ValueError): validate_archive(path)

    def test_session_restores_paused_and_saves_only_after_quit(self):
        from minecraftrl.world_store import WorldStore
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.tasks import CommandBus
        from test_task_session import FakeEnvironment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); world,task,obs=self.fixture(root)
            class Environment(FakeEnvironment):
                def observation(self): return dict(super().observation(),location_stats=obs['location_stats'])
                def save_and_quit(self): self.saved=True; return world
            class Policy:
                def act(self,frame): raise AssertionError('restored paused task dispatched')
                def close(self): pass
            env=Environment(); env.saved=False
            with WorldStore(root/'store','companion') as store:
                store.commit(world,task,obs); bundle=store.load()
                report=run_task_session(Policy(),env,root/'output',seed=1,limit_s=.04,
                    bus=CommandBus(),save_screen=lambda *a:None,world_store=store,restore_bundle=bundle)
                self.assertTrue(env.saved)
                self.assertEqual(report['task']['status'],'paused')
                self.assertEqual(report['task']['progress'],2)
                self.assertEqual(report['world_save']['status'],'saved')
                self.assertEqual(store.load()['world_id'],'world-a')

    def test_restore_mismatch_does_not_publish_save_or_dispatch(self):
        from minecraftrl.world_store import WorldStore
        from minecraftrl.task_runtime import run_task_session
        from minecraftrl.tasks import CommandBus
        from test_task_session import FakeEnvironment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); world,task,obs=self.fixture(root)
            class Environment(FakeEnvironment):
                def observation(self): return dict(super().observation(),location_stats=obs['location_stats'])
                def save_and_quit(self): raise AssertionError('invalid restored world saved')
            class Policy:
                def close(self): pass
            env=Environment(); env.inventory={}
            with WorldStore(root/'store','companion') as store:
                store.commit(world,task,obs); bundle=store.load()
                report=run_task_session(Policy(),env,root/'output',seed=1,limit_s=.04,
                    bus=CommandBus(),save_screen=lambda *a:None,world_store=store,restore_bundle=bundle)
                self.assertEqual(report['status'],'error')
                self.assertEqual(report['steps'],0)
                self.assertEqual(store.load()['generation'],bundle['generation'])


if __name__=='__main__': unittest.main()
