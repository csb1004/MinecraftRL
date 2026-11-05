import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from minecraftrl.workshop import WorkshopEnvironment,WorkshopController,recipe_selection,table_aim_points
from minecraftrl.workshop_recipes import RecipeCatalog


class WorkshopPersistenceTests(unittest.TestCase):
    def test_owned_food_consumption_requires_count_and_hunger_change(self):
        for consume,increase,success in ((True,True,True),(True,False,False),(False,True,False)):
            stock={'stone_axe':1,'apple':1,'oak_log':103}
            c=WorkshopController(None,stock,['stone_axe'],Path('.'),equip='stone_axe',equip_only=True,equip_slot=1,consume_food='apple')
            gui=dict(error=0,server_available=1,server_alive=1,server_food=0,server_health=4.166666,
                cursor_count=0,server_cursor=0,server_inventory=dict(stock),server_selected_item='apple',screen=0)
            c.environment=SimpleNamespace(gui=gui);c.noop=lambda:{};c.mark=Mock();c.equip_tool=lambda:iter(())
            seq=c.consume_owned_food();self.assertEqual(next(seq),{'use':1})
            if consume:gui['server_inventory'].pop('apple')
            if increase:gui['server_food']=4
            if success:
                list(seq);self.assertEqual(+c.expected,{'stone_axe':1,'oak_log':103})
                self.assertEqual(c.consumption['quantity'],1);self.assertEqual(c.consumption['final_food'],4)
            else:
                with self.assertRaises(ValueError):list(seq)
            self.assertEqual(c.equip,'stone_axe');self.assertEqual(c.equip_slot,1)

    def test_food_assistance_cannot_invent_food_or_enable_unsupported_consumables(self):
        for stock,food,equip_only in (({'stone_axe':1},'apple',True),
            ({'stone_axe':1,'apple':1},'apple',False),({'stone_axe':1,'dirt':1},'dirt',True)):
            with self.assertRaises(ValueError):WorkshopController(None,stock,['stone_axe'],Path('.'),
                equip='stone_axe',equip_only=equip_only,consume_food=food)

    def test_tool_relocation_swaps_an_occupied_slot_without_consuming_inventory(self):
        stock={'stone_axe':1,'oak_log':64}
        c=WorkshopController(None,stock,['stone_axe'],Path('.'),equip='stone_axe',equip_only=True,equip_slot=1)
        gui=dict(error=0,server_available=1,cursor_item='air',cursor_count=0,server_cursor=0,server_inventory=dict(stock),
            slots=[dict(inventory_index=0,item='oak_log',count=64),dict(inventory_index=3,item='stone_axe',count=1)])
        c.environment=SimpleNamespace(gui=gui);c.noop=lambda:{};c.mark=Mock();clicks=[]
        def click(index):
            clicks.append(index);slot=gui['slots'][index]
            slot['item'],gui['cursor_item']=gui['cursor_item'],slot['item']
            slot['count'],gui['cursor_count']=gui['cursor_count'],slot['count']
            gui['server_cursor']=gui['cursor_count']
            gui['server_inventory']={s['item']:s['count'] for s in gui['slots'] if s['count']}
            yield {'click':index}
        c.slot_click=click
        list(c.relocate_tool(1,0))
        self.assertEqual(clicks,[1,0,1]);self.assertTrue(c.committed())
        self.assertEqual(gui['slots'][0]['item'],'stone_axe');self.assertEqual(gui['slots'][1]['item'],'oak_log')
        self.assertEqual(gui['server_inventory'],stock)
        list(c.relocate_tool(0,0));self.assertEqual(clicks,[1,0,1])

    def test_tool_slot_requires_an_explicit_tool_and_valid_hotbar_index(self):
        for slot in (0,10,True):
            with self.assertRaises(ValueError):
                WorkshopController(None,{'stone_axe':1},['stone_axe'],Path('.'),equip='stone_axe',equip_slot=slot)
        with self.assertRaises(ValueError):WorkshopController(None,{},[],Path('.'),equip_slot=1)

    def test_equipment_only_neither_plans_nor_requires_a_workstation(self):
        catalog=Mock();stock={'stone_axe':1,'oak_log':76}
        controller=WorkshopController(catalog,stock,['stone_axe'],Path('.'),equip='stone_axe',equip_only=True)
        controller.noop=Mock(return_value={});controller.committed=Mock(return_value=True)
        controller.equip_tool=Mock(return_value=iter([{'hotbar.4':1}]))
        seq=controller.actions();self.assertEqual(next(seq),{});self.assertEqual(next(seq),{'hotbar.4':1});next(seq)
        catalog.plan.assert_not_called();self.assertEqual(controller.expected,stock)
        raw=Mock();raw.step.return_value=({},0,False,{'workshop':dict(screen=0,server_inventory=stock,server_selected_item='stone_axe')})
        controller.mark=Mock();wrapped=WorkshopEnvironment(raw,controller,{},Mock(),Path('.'))
        wrapped.step({});self.assertTrue(wrapped.completed);self.assertIsNone(controller.table)
        with self.assertRaises(RuntimeError):wrapped.step({})

    def test_equipment_only_cannot_invent_a_tool_or_accept_wrong_selected_item(self):
        with self.assertRaises(ValueError):WorkshopController(None,{},['stone_axe'],Path('.'),equip='stone_axe',equip_only=True)
        c=WorkshopController(None,{'stone_axe':1},['stone_axe'],Path('.'),equip='stone_axe',equip_only=True)
        c.stage='finish';c.committed=Mock(return_value=True)
        raw=Mock();raw.step.return_value=({},0,False,{'workshop':dict(screen=0,server_inventory={'stone_axe':1},server_selected_item='oak_log')})
        with self.assertRaisesRegex(ValueError,'equip requirement'):WorkshopEnvironment(raw,c,{},Mock(),Path('.')).step({})

    def test_workstation_targets_include_near_side_under_overhead_occlusion(self):
        points=table_aim_points((211,71,-827),(210.6069,72.62,-826.2301))
        self.assertEqual(len(points),7)
        self.assertIn((211.001,71.5,-826.5),points)
        self.assertTrue(all(211<x<212 and 71<y<72 and -827<z<-826 for x,y,z in points))

    def test_group_requires_right_click_then_exact_visible_overlay_selection(self):
        recipe=dict(known=1,server_known=1,craftable=1,single=0,x=-1,y=-1,group_x=40,group_y=60,overlay=0)
        self.assertEqual(recipe_selection(recipe),('use',40,60))
        self.assertEqual(recipe_selection(dict(recipe,overlay=1,x=84,y=92)),('attack',84,92))
        with self.assertRaises(ValueError):recipe_selection(dict(recipe,group_x=-1,x=84,y=92))
        with self.assertRaises(ValueError):recipe_selection(dict(recipe,server_known=0))

    def test_gui_save_preserves_goal_history_and_invalidates_unobserved_neural_memory(self):
        checkpoint={'schema':4,'world_id':'test','goals':{'wood':{'task':{'status':'completed','progress':64}}}}
        bundle={'world_id':'test','task_checkpoint':copy.deepcopy(checkpoint),
                'controller_state':{'recovery':{'failed':['old-method']}},'policy_memory':{'old':True}}
        env=Mock();env.save_and_quit.return_value=Path('saved-world')
        store=Mock();store.commit.return_value=Path('new-generation')
        controller=SimpleNamespace(goals=['wooden_pickaxe'],milestones=[{'event':'recipe_committed'}])
        wrapper=WorkshopEnvironment(env,controller,bundle,store,Path('.'))
        wrapper.restore={'inventory_match':True};wrapper.completed=True;wrapper.steps=40
        wrapper.close()
        args,kwargs=store.commit.call_args
        self.assertEqual(args[1],checkpoint)
        self.assertEqual(kwargs['controller_state']['recovery'],{'failed':['old-method']})
        self.assertNotIn('policy_memory',kwargs)
        self.assertIn('policy_memory_invalidated',kwargs['controller_state']['last_workshop'])
        self.assertNotIn('last_workshop',bundle['controller_state'])
        env.close.assert_called_once()

    def test_existing_workstation_is_not_charged_twice_and_stone_is_required(self):
        recipes={'stone_axe':{'type':'minecraft:crafting_shaped','pattern':['CC','CS',' S'],
            'key':{'C':{'item':'minecraft:cobblestone'},'S':{'item':'minecraft:stick'}},
            'result':{'item':'minecraft:stone_axe'}}}
        catalog=RecipeCatalog(recipes,{})
        stock={'cobblestone':3,'stick':2,'oak_log':64}
        plan,left=catalog.plan(stock,['stone_axe'],workstation_available=True)
        self.assertEqual(plan,['stone_axe']);self.assertEqual(left,{'stone_axe':1,'oak_log':64})
        with self.assertRaises(ValueError):catalog.plan(dict(stock,cobblestone=2),['stone_axe'],workstation_available=True)
        self.assertEqual(stock['cobblestone'],3)
