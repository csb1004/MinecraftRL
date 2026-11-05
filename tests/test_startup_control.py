import copy
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from minecraftrl.runtime import write_json
from minecraftrl.startup_control import (
    ENVIRONMENT_KEY,configure_startup_control,prepare_startup_world,verify_startup_receipt,verify_world_pair,
)
from minecraftrl.world_store import digest

try:
    import nbtlib as n
except ImportError:
    n=None


@unittest.skipUnless(n is not None,'offline NBT preparation uses isolated fixture-tools')
class StartupWorldTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source.zip'
        self.output=self.root/'paused.zip';self.run=self.root/'run';self.run.mkdir()
        level=n.File({'Data':n.Compound(GameRules=n.Compound(randomTickSpeed=n.String('3'),doMobSpawning=n.String('true')),
            WorldGenSettings=n.Compound(seed=n.Long(104743)),Time=n.Long(44),
            Player=n.Compound(Health=n.Float(20),Pos=n.List[n.Double]([930.5,66.,554.5])))})
        buffer=io.BytesIO();level.write(buffer)
        with zipfile.ZipFile(self.source,'x') as archive:
            archive.writestr('./saves/test/level.dat',gzip.compress(buffer.getvalue(),mtime=0))
            archive.writestr('./saves/test/region/r.1.1.mca',b'preserved terrain bytes')
        self.meta=dict(seed=104743,experimental_split='train',sha256=digest(self.source),
            kind='natural_navigation',navigation_goals={'source_archive_sha256':digest(self.source),'points':[[1,2,3]]})
        write_json(self.source.with_suffix('.json'),self.meta)
        self.addCleanup(os.environ.pop,ENVIRONMENT_KEY,None)

    def prepare(self):
        return prepare_startup_world(self.root,self.source,self.output)

    def test_copy_preserves_source_and_only_changes_the_one_game_rule(self):
        before=self.source.read_bytes();manifest=self.prepare()
        self.assertEqual(self.source.read_bytes(),before)
        verify_world_pair(self.source,self.output,3)
        self.assertEqual(manifest['navigation_goals']['points'],self.meta['navigation_goals']['points'])
        self.assertEqual(manifest['navigation_goals']['source_archive_sha256'],digest(self.output))
        control=configure_startup_control(self.root,self.output,manifest,self.run,enabled=True)
        self.assertEqual(control['restore_speed'],3)
        self.assertEqual(os.environ[ENVIRONMENT_KEY],control['request_path'])
        with self.assertRaises(FileExistsError):self.prepare()
        with self.assertRaises(FileExistsError):configure_startup_control(self.root,self.output,manifest,self.run,enabled=True)

    def test_controlled_world_cannot_run_without_explicit_flag(self):
        manifest=self.prepare()
        with self.assertRaises(ValueError):configure_startup_control(self.root,self.output,manifest,self.run)
        with self.assertRaises(ValueError):configure_startup_control(self.root,self.source,self.meta,self.run,enabled=True)
        self.assertFalse((self.run/'startup-random-ticks-request.json').exists())

    def test_split_seed_and_boundary_tampering_are_rejected(self):
        manifest=self.prepare()
        cases=[]
        bad=copy.deepcopy(manifest);bad['experimental_split']='final';cases.append(bad)
        bad=copy.deepcopy(manifest);bad['seed']=12;cases.append(bad)
        for key,value in (('resume_client_tick',81),('startup_steps',61),('restore_speed',True),('paused_speed',False)):
            bad=copy.deepcopy(manifest);bad['startup_control'][key]=value;cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                configure_startup_control(self.root,self.output,bad,self.run,enabled=True)
        self.assertNotIn(ENVIRONMENT_KEY,os.environ)

    def test_other_member_mutation_is_rejected_even_with_recomputed_derived_hash(self):
        manifest=self.prepare();bad=self.root/'bad.zip'
        with zipfile.ZipFile(self.output) as original,zipfile.ZipFile(bad,'x') as target:
            for entry in original.infolist():
                value=b'changed terrain' if entry.filename.endswith('.mca') else original.read(entry)
                target.writestr(copy.copy(entry),value)
        manifest['sha256']=digest(bad)
        with self.assertRaisesRegex(ValueError,'changed world member'):
            configure_startup_control(self.root,bad,manifest,self.run,enabled=True)

    def test_final_or_wrong_saved_seed_is_rejected_before_output(self):
        for change in ({'experimental_split':'final'},{'seed':10}):
            write_json(self.source.with_suffix('.json'),dict(self.meta,**change))
            with self.assertRaises(ValueError):self.prepare()
            self.assertFalse(self.output.exists())


class StartupReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        request=self.root/'request.json';write_json(request,{'request':'frozen'})
        self.control=dict(request_path=str(request),request_sha256=digest(request),output=str(self.root),
            restore_speed=3,resume_client_tick=80)
        self.server=dict(client_tick=80.,server_tick=82.,world_time=126.)
        self.receipt=dict(schema='startup_random_tick_receipt_v1',restored=True,
            request_sha256=digest(request),restore_speed=3,current_speed=3,
            client_tick=80,client_tick_before=80,server_tick=82,server_tick_before=82,
            world_time=126,world_time_before=126,extra_simulation_ticks=0,input_commands=0,
            random_state_before=1234,random_state_after=1234)

    def save(self,value):
        write_json(self.root/'startup-random-ticks-restored.json',value)

    def test_missing_or_unbound_receipt_fails_before_control(self):
        with self.assertRaisesRegex(ValueError,'unavailable'):verify_startup_receipt(self.control,self.server)
        for change in ({'restored':False},{'request_sha256':'0'*64}):
            self.save(dict(self.receipt,**change))
            with self.assertRaises(ValueError):verify_startup_receipt(self.control,self.server)

    def test_correct_restore_and_next_actual_capture_are_separate(self):
        self.save(self.receipt)
        self.assertTrue(verify_startup_receipt(self.control,self.server)['verified'])
        next_receipt=dict(self.receipt,client_tick=81,client_tick_before=81,server_tick=83,server_tick_before=83,
            world_time=127,world_time_before=127)
        write_json(self.root/'startup-random-ticks-first-step.json',next_receipt)
        self.assertTrue(verify_startup_receipt(self.control,dict(client_tick=81.,server_tick=83.,world_time=127.),first_step=True)['verified'])
        with self.assertRaises(ValueError):verify_startup_receipt(self.control,self.server,first_step=True)

    def test_wrong_rule_tick_rng_or_extra_input_invalidates_receipt(self):
        changes=({'current_speed':0},{'client_tick':81},{'server_tick_before':81},{'world_time':125},
            {'random_state_after':1235},{'random_state_before':True,'random_state_after':True},
            {'extra_simulation_ticks':1},{'input_commands':False})
        for change in changes:
            self.save(dict(self.receipt,**change))
            with self.subTest(change=change),self.assertRaises(ValueError):verify_startup_receipt(self.control,self.server)

    def test_changed_request_and_ambient_activation_are_rejected(self):
        self.save(self.receipt)
        Path(self.control['request_path']).write_text('modified')
        with self.assertRaises(ValueError):verify_startup_receipt(self.control,self.server)
        with patch.dict(os.environ,{ENVIRONMENT_KEY:'unrequested'}):
            self.assertIsNone(configure_startup_control(self.root,None,None,self.root,enabled=False))
            self.assertNotIn(ENVIRONMENT_KEY,os.environ)
        self.assertIsNone(verify_startup_receipt(None,{}))


if __name__=='__main__':unittest.main()
