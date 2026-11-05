import gzip
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from minecraftrl.saved_inventory import verify_saved_inventory,verify_saved_equipment,verify_saved_food


def text(value):
    value=value.encode();return struct.pack('>H',len(value))+value
def inventory(counts):
    entries=[]
    for count in counts:
        entries.append(b'\x08'+text('id')+text('minecraft:oak_planks')+b'\x01'+text('Count')+struct.pack('b',count)+b'\0')
    return b'\x09'+text('Inventory')+b'\x0a'+struct.pack('>i',len(entries))+b''.join(entries)+b'\0'


class SavedInventoryTests(unittest.TestCase):
    def test_food_recovery_requires_both_saved_player_records(self):
        def player(food):return b'\x03'+text('foodLevel')+struct.pack('>i',food)+b'\x00'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'world.zip'
            for food in (0,4):
                with zipfile.ZipFile(path,'w') as archive:
                    archive.writestr('./saves/world/level.dat',gzip.compress(b'\x0a'+text('')+b'\x0a'+text('Data')+b'\x0a'+text('Player')+player(4)+b'\x00\x00'))
                    archive.writestr('./saves/world/playerdata/player.dat',gzip.compress(b'\x0a'+text('')+player(food)))
                self.assertEqual(verify_saved_food(path,4)['verified'],food==4)

    def test_selected_tool_requires_matching_level_and_player_hotbar_records(self):
        def player(selected):
            item=b'\x08'+text('id')+text('minecraft:stone_axe')+b'\x01'+text('Count')+b'\x01'+b'\x01'+text('Slot')+b'\x00'+b'\x00'
            return b'\x03'+text('SelectedItemSlot')+struct.pack('>i',selected)+b'\x09'+text('Inventory')+b'\x0a'+struct.pack('>i',1)+item+b'\x00'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'world.zip'
            for selected in (0,3):
                with zipfile.ZipFile(path,'w') as archive:
                    archive.writestr('./saves/world/level.dat',gzip.compress(b'\x0a'+text('')+b'\x0a'+text('Data')+b'\x0a'+text('Player')+player(0)+b'\x00\x00'))
                    archive.writestr('./saves/world/playerdata/player.dat',gzip.compress(b'\x0a'+text('')+player(selected)))
                self.assertEqual(verify_saved_equipment(path,'stone_axe',1)['verified'],selected==0)
                self.assertFalse(verify_saved_equipment(path,'wooden_pickaxe',1)['verified'])

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
