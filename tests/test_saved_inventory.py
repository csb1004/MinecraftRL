import gzip
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from minecraftrl.saved_inventory import verify_saved_inventory


def text(value):
    value=value.encode();return struct.pack('>H',len(value))+value
def inventory(counts):
    entries=[]
    for count in counts:
        entries.append(b'\x08'+text('id')+text('minecraft:oak_planks')+b'\x01'+text('Count')+struct.pack('b',count)+b'\0')
    return b'\x09'+text('Inventory')+b'\x0a'+struct.pack('>i',len(entries))+b''.join(entries)+b'\0'


class SavedInventoryTests(unittest.TestCase):
    def check(self,counts,expected):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'world.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('./saves/world/level.dat',gzip.compress(b'\x0a'+text('')+b'\x0a'+text('Data')+b'\x0a'+text('Player')+inventory(counts)+b'\0\0'))
                archive.writestr('./saves/world/playerdata/player.dat',gzip.compress(b'\x0a'+text('')+inventory(counts)))
            return verify_saved_inventory(path,expected)
    def test_sums_duplicate_slots(self):
        self.assertTrue(self.check([1,3],{'oak_planks':4})['verified'])
    def test_rejects_client_prediction_absent_from_saved_inventory(self):
        self.assertFalse(self.check([],{'oak_planks':4})['verified'])
