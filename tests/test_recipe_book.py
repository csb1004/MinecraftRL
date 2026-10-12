import unittest
from minecraftrl.recipe_book import verify_grid, require_recipe_button


class RecipeBookTests(unittest.TestCase):
    def gui(self, counts=(1,0,1,0)):
        g={'preview_item':2,'preview_count':4,'oak_planks_id':1,'stick_id':2,'crafting_table_id':3,'cursor_count':0}
        for i,n in enumerate(counts,1):g.update({f'input_{i}_item':1 if n else 0,f'input_{i}_count':n})
        return g

    def test_sticks_accept_either_vertical_column(self):
        for counts in ((1,0,1,0),(0,1,0,1)):verify_grid(self.gui(counts),'sticks')

    def test_preview_alone_and_excess_material_are_rejected(self):
        for counts in ((0,0,0,0),(1,1,0,0),(2,0,2,0),(1,1,1,1)):
            with self.assertRaises(ValueError):verify_grid(self.gui(counts),'sticks')

    def test_wrong_item_and_wrong_result_are_rejected(self):
        for field,value in (('input_1_item',9),('preview_item',3),('cursor_count',4)):
            g=self.gui();g[field]=value
            with self.assertRaises(ValueError):verify_grid(g,'sticks')

    def test_table_requires_four_single_planks_and_matching_result(self):
        g=self.gui((1,1,1,1));g.update(preview_item=3,preview_count=1)
        verify_grid(g,'table')

    def test_recipe_must_be_unlocked_visible_single_and_craftable(self):
        g={'known_stick':1,'server_known_stick':1,'book_stick_x':50,'book_stick_y':60,
           'book_stick_single':1,'book_stick_craftable':1,'book_open':1,'book_error':0}
        self.assertEqual(require_recipe_button(g,'stick'),(50,60))
        for key,value in (('server_known_stick',0),('book_stick_x',-1),('book_stick_single',0),
                          ('book_stick_craftable',0),('book_open',0),('book_error',1)):
            invalid=dict(g);invalid[key]=value
            with self.assertRaises(ValueError):require_recipe_button(invalid,'stick')
