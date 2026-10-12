import unittest
from minecraftrl.saved_block import section_block


class SavedBlockTests(unittest.TestCase):
    def test_padded_longs_and_negative_coordinates(self):
        palette=[{'Name':'block'+str(i)} for i in range(17)]
        words=[0]*342
        index=15*256+15*16+15
        words[index//12]=16<<((index%12)*5)
        section={'Palette':palette,'BlockStates':words}
        self.assertEqual(section_block(section,-1,-1,-1),'block16')
        self.assertEqual(section_block(section,0,0,0),'block0')

    def test_uniform_section(self):
        self.assertEqual(section_block({'Palette':[{'Name':'minecraft:air'}]},0,0,0),'minecraft:air')
