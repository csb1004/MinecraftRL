import gzip
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
import zlib

from minecraftrl.startup_pickups import saved_drop_budget,validate_pickup_delta,eligible_drop_position
from minecraftrl.world_store import validate_restored_observation


def text(value):
    data=value.encode();return struct.pack('>H',len(data))+data


def tag(kind,name,payload):return bytes([kind])+text(name)+payload
def compound(**fields):return b''.join(tag(kind,name,value) for name,(kind,value) in fields.items())+b'\0'


def fixture(path):
    item=compound(id=(8,text('minecraft:dirt')),Count=(1,b'\x08'))
    inventory=tag(9,'Inventory',b'\x0a'+struct.pack('>i',1)+item)
    pos=tag(9,'Pos',b'\x06'+struct.pack('>i',3)+struct.pack('>ddd',0.,64.,0.))
    player=inventory+pos+b'\0'
    drop=compound(id=(8,text('minecraft:item')),Pos=(9,b'\x06'+struct.pack('>i',3)+struct.pack('>ddd',0.5,65.,0.)),
        Item=(10,compound(id=(8,text('minecraft:dirt')),Count=(1,b'\x01'))),
        PickupDelay=(2,struct.pack('>h',5)),Age=(2,struct.pack('>h',5)),UUID=(11,struct.pack('>iiiii',4,1,2,3,4)))
    level=compound(xPos=(3,struct.pack('>i',0)),zPos=(3,struct.pack('>i',0)),Entities=(9,b'\x0a'+struct.pack('>i',1)+drop))
    chunk=tag(10,'',compound(DataVersion=(3,struct.pack('>i',2586)),Level=(10,level)))
    payload=zlib.compress(chunk); region=bytearray(8192);region[:4]=b'\0\0\x02\x01'
    region.extend(struct.pack('>i',len(payload)+1)+b'\x02'+payload)
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('./saves/world/level.dat',gzip.compress(tag(10,'',tag(10,'Data',tag(10,'Player',player)+b'\0')+b'\0')))
        archive.writestr('./saves/world/playerdata/player.dat',gzip.compress(tag(10,'',player)))
        archive.writestr('./saves/world/region/r.0.0.mca',region)


class StartupPickupTests(unittest.TestCase):
    def test_falling_drop_requires_explicit_bounded_column(self):
        player=[186.7,70.,-822.04];item={'Pos':[186.375,73.75,-822.36],'Motion':[.065,0.,.021]}
        self.assertFalse(eligible_drop_position(player,item))
        self.assertTrue(eligible_drop_position(player,item,include_falling=True))
        for change in ({'Pos':[190.,73.75,-822.]},{'Pos':[186.7,77.,-822.]},
                       {'NoGravity':True},{'Motion':[1.,0.,0.]},{'Motion':[0.,float('nan'),0.]},{'Motion':[]}):
            self.assertFalse(eligible_drop_position(player,dict(item,**change),include_falling=True))

    def test_actual_nbt_decoder_binds_drop_budget_to_saved_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'world.zip';fixture(archive)
            evidence=saved_drop_budget(archive,{'dirt':8})
            self.assertEqual(evidence['inventory_budget'],{'dirt':1})
            self.assertFalse(evidence['policy_action_credit'])
            with self.assertRaises(ValueError):saved_drop_budget(archive,{'dirt':7})
        self.assertTrue(validate_pickup_delta({'dirt':8},{'dirt':9},evidence)['consistent_with_saved_drops'])

    def test_losses_unknown_items_and_excess_counts_remain_rejected(self):
        evidence={'inventory_budget':{'dirt':1}}
        for actual in ({'dirt':7},{'dirt':10},{'dirt':8,'oak_log':1},{'dirt':8}):
            with self.subTest(actual=actual),self.assertRaises(ValueError):validate_pickup_delta({'dirt':8},actual,evidence)

    def test_strict_default_remains_strict_and_opt_in_is_not_an_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'world.zip';fixture(archive)
            obs=dict(inventory={'dirt':8},location_stats=dict(xpos=0.,ypos=64.,zpos=0.))
            bundle=dict(archive=str(archive),observation=obs)
            actual=dict(obs,inventory={'dirt':9})
            with self.assertRaisesRegex(ValueError,'restored_inventory_mismatch'):validate_restored_observation(bundle,actual)
            result=validate_restored_observation(bundle,actual,allow_startup_pickups=True)
            self.assertFalse(result['inventory_match'])
            self.assertEqual(result['startup_pickups']['inventory_delta'],{'dirt':1})
            with self.assertRaisesRegex(ValueError,'position'):
                validate_restored_observation(bundle,dict(actual,location_stats=dict(xpos=20.,ypos=64.,zpos=0.)),allow_startup_pickups=True)


if __name__=='__main__': unittest.main()
