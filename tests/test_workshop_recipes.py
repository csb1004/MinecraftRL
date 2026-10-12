import unittest
from minecraftrl.workshop_recipes import RecipeCatalog


class WorkshopRecipeTests(unittest.TestCase):
    def catalog(self):
        recipes={
            'planks':{'type':'minecraft:crafting_shapeless','ingredients':[{'item':'minecraft:oak_log'}],'result':{'item':'minecraft:oak_planks','count':4}},
            'table':{'type':'minecraft:crafting_shaped','pattern':['PP','PP'],'key':{'P':{'tag':'minecraft:planks'}},'result':{'item':'minecraft:crafting_table'}},
            'sticks':{'type':'minecraft:crafting_shaped','pattern':['P','P'],'key':{'P':{'tag':'minecraft:planks'}},'result':{'item':'minecraft:stick','count':4}},
            'pick':{'type':'minecraft:crafting_shaped','pattern':['PPP',' S ',' S '],'key':{'P':{'tag':'minecraft:planks'},'S':{'item':'minecraft:stick'}},'result':{'item':'minecraft:wooden_pickaxe'}},
            'axe':{'type':'minecraft:crafting_shaped','pattern':['PP','PS',' S'],'key':{'P':{'tag':'minecraft:planks'},'S':{'item':'minecraft:stick'}},'result':{'item':'minecraft:wooden_axe'}}}
        return RecipeCatalog(recipes,{'minecraft:planks':['minecraft:oak_planks','minecraft:birch_planks']})

    def test_plan_reserves_shared_materials_and_accounts_for_batch_outputs(self):
        plan,remaining=self.catalog().plan({'oak_log':3},['wooden_pickaxe','wooden_axe'])
        self.assertEqual(remaining,{'crafting_table':1,'wooden_pickaxe':1,'wooden_axe':1})
        self.assertEqual(plan.count('planks'),3);self.assertEqual(plan.count('sticks'),1)
        self.assertLess(plan.index('table'),plan.index('pick'))

    def test_insufficient_resources_rejected_before_execution(self):
        with self.assertRaises(ValueError):self.catalog().plan({'oak_log':2},['wooden_pickaxe','wooden_axe'])

    def test_axe_mirror_translation_and_tag_substitution(self):
        c=self.catalog();items=['air','birch_planks','birch_planks','air','stick','birch_planks','air','stick','air']
        self.assertTrue(c.matches('axe',3,items,[int(x!='air') for x in items]))
        self.assertFalse(c.matches('axe',2,items[:4],[1]*4))
        self.assertFalse(c.matches('axe',3,items,[2 if x!='air' else 0 for x in items]))

    def test_wrong_preview_empty_grid_and_extra_ingredient_rejected(self):
        c=self.catalog()
        self.assertFalse(c.matches('sticks',2,['oak_planks','oak_planks','air','air'],[1,1,0,0]))
        self.assertFalse(c.matches('sticks',2,['oak_planks','air','oak_planks','stick'],[1,0,1,1]))
        self.assertTrue(c.matches('planks',2,['air','oak_log','air','air'],[0,1,0,0]))
