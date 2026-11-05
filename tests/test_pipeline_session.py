import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minecraftrl.pipeline.recorder import Recorder
from minecraftrl.pipeline.session import DemoSession, fixture_sha52, set_condition
from minecraftrl.pipeline.steps import Step
from minecraftrl.runtime import StopGate

SKILLS = {'dig': dict(embedding=np.full((1, 512), .5, np.float32), cond_scale=7., sha256='d'),
          'wood': dict(embedding=np.full((1, 512), .25, np.float32), cond_scale=6., sha256='w')}
KEYS = ('attack', 'use', 'forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'inventory', 'drop',
        *('hotbar.%d' % i for i in range(1, 10)))


def noop():
    action = {key: 0 for key in KEYS}; action['camera'] = [0., 0.]; return action


class FakeRaw:
    class action_space:
        noop = staticmethod(noop)


def info(inventory=None, *, fixture_state=2, sha52=0, screen=0, selected='air', table=(0, 0, 0),
         mining='air', hit='air', destroys=0, tick=1000, x=0.5, y=64., z=0.5, pitch=0., **demo):
    workshop = dict(server_available=1, error=0, screen=screen, server_selected_item=selected, server_x=x, server_y=y, server_z=z, pitch=pitch,
                    server_inventory=dict(inventory or {}), world_time=tick, server_tick=tick,
                    table_x=table[0], table_y=table[1], table_z=table[2], cursor_count=0, server_cursor=0)
    values = dict(demo_available=1, demo_fixture_state=fixture_state, demo_fixture_sha52=sha52,
                  demo_mining_block=mining, demo_hit_block=hit, demo_destroys=destroys)
    values.update(demo)
    return {'workshop': workshop, 'demo': values}


class FakeEnv:
    def __init__(self, script, *, stop_file=None, stop_at=None, done_at=None):
        self.raw = FakeRaw(); self.script = script; self.calls = 0; self.actions = []; self.released = 0
        self.stop_file = stop_file; self.stop_at = stop_at; self.done_at = done_at
    def reset(self):
        return {'pov': np.zeros((4, 4, 3), np.uint8)}
    def step(self, action):
        self.calls += 1; self.actions.append(action)
        if self.stop_at == self.calls: self.stop_file.touch()
        return {'pov': np.zeros((4, 4, 3), np.uint8)}, 0., self.done_at == self.calls, self.script(self.calls, action)
    def release(self):
        # Same contract as policies.MineRLEnvironment.release after a terminal step (e.g. death).
        self.released += 1
        return 'environment_done_no_neutral_step' if self.done_at is not None and self.calls >= self.done_at else 'neutral_command_sent'


class FakeAgent:
    def __init__(self): self.scales = []
    def reset(self, cond_scale): self.scales.append(cond_scale)


class FakePolicy:
    def __init__(self, action=None):
        self.agent = FakeAgent(); self.goal_embed = None; self.action = action or noop(); self.calls = 0
    def act(self, screen):
        if self.goal_embed is None: raise RuntimeError('condition required')
        self.calls += 1; return dict(self.action)


class FakeController:
    """Workshop generator stand-in: three actions, then the finish stage."""
    created = []
    def __init__(self, catalog, inventory, goals, output, *, equip=None, **kw):
        self.goals = goals; self.equip = equip; self.kw = kw; self.inventory = dict(inventory); self.stage = 'initializing'
        self.table = (1, 2, 3); FakeController.created.append(self)
        self.environment = None; self.milestones = []
    def reset(self, goal=None): self.sequence = self.actions()
    def committed(self): return True
    def actions(self):
        for _ in range(3):
            action = self.environment.env.raw.action_space.noop(); action['use'] = 1; yield action
        self.stage = 'finish'
        while True: yield self.environment.env.raw.action_space.noop()


class FailingController(FakeController):
    def actions(self):
        yield self.environment.env.raw.action_space.noop()
        raise ValueError('workshop timeout: open 2x2 inventory')


class Clock:
    def __init__(self, step=0.): self.now = 0.; self.step = step
    def __call__(self):
        self.now += self.step; return self.now


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.out = Path(self.temp.name)
        self.stop = self.out / 'STOP'; self.gate = StopGate()

    def tearDown(self):
        if hasattr(self, 'recorder'): self.recorder.close()
        self.temp.cleanup()

    def session(self, env, policy=None, *, limit_s=60., clock=None, controller=FakeController, fixture=None, tool_lock=False):
        self.recorder = Recorder(self.out / 'run')
        return DemoSession(env, policy or FakePolicy(), None, self.recorder, self.gate, self.stop,
                           limit_s=limit_s, skills=SKILLS, clock=clock or Clock(), rule_controller=controller,
                           fixture_sha52=fixture, settle_steps=0, tool_lock=tool_lock)

    def timeline(self):
        return [json.loads(line) for line in (self.out / 'run/timeline.jsonl').read_text().splitlines()]

    def test_stop_file_prevents_next_dispatch(self):
        env = FakeEnv(lambda n, a: info(), stop_file=self.stop, stop_at=4)
        report = self.session(env).run([Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual(env.calls, 4)
        self.assertEqual(env.released, 1)
        self.assertEqual(report['status'], 'stop_file')
        self.assertFalse(report['all_steps_completed'])
        kinds = [line['kind'] for line in self.timeline()]
        self.assertLess(kinds.index('stop_acknowledged'), kinds.index('input_release'))

    def test_death_is_a_valid_policy_failure(self):
        # Real probe: the policy swam into lava; the episode ended and there was no neutral step to send.
        env = FakeEnv(lambda n, a: info(), done_at=5)
        report = self.session(env).run([Step('s1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual(report['status'], 'environment_done')
        self.assertEqual(report['release'], 'environment_done_no_neutral_step')
        self.assertTrue(report['execution_valid'])
        self.assertFalse(report['all_steps_completed'])
        self.assertEqual(env.calls, 5)

    def test_terminal_release_before_done_is_invalid(self):
        class EarlyTerminal(FakeEnv):
            def release(self): self.released += 1; return 'environment_done_no_neutral_step'
        env = EarlyTerminal(lambda n, a: info(), stop_file=self.stop, stop_at=4)
        report = self.session(env).run([Step('s1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual(report['status'], 'stop_file')
        self.assertFalse(report['execution_valid'])

    def test_policy_step_completes_on_server_inventory_gain(self):
        env = FakeEnv(lambda n, a: info({'cobblestone': 3 if n >= 5 else 0, 'wooden_pickaxe': 1}))
        policy = FakePolicy()
        report = self.session(env, policy).run([Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'completed'); self.assertEqual(stage['controller'], 'policy')
        self.assertEqual(stage['gain'], 3)
        self.assertEqual(report['status'], 'completed'); self.assertTrue(report['all_steps_completed'])
        self.assertIs(policy.goal_embed, SKILLS['dig']['embedding']); self.assertEqual(policy.agent.scales, [7.])
        self.assertEqual(report['final_inventory'], {'cobblestone': 3, 'wooden_pickaxe': 1})
        self.assertEqual(env.released, 1)

    def test_tool_lock_masks_policy_slot_switch(self):
        action = noop(); action['hotbar.3'] = 1; action['forward'] = 1
        env = FakeEnv(lambda n, a: info({'cobblestone': 3 if n >= 4 else 0}, selected='wooden_pickaxe'))
        report = self.session(env, FakePolicy(action), tool_lock=True).run(
            [Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        policy_actions = env.actions[1:]
        self.assertEqual([(a['hotbar.3'], a['forward']) for a in policy_actions], [(0, 1)] * 3)
        self.assertEqual(report['stage_results'][0]['tool_masked'], 3)
        self.assertTrue(report['tool_lock'])
        telemetry = [json.loads(l) for l in (self.out / 'run/telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([t.get('masked') for t in telemetry if t['controller'] == 'policy'], [['hotbar.3']] * 3)

    def test_tool_lock_masks_drop_and_inventory(self):
        # Real probe: STEVE-1 dropped the given stone pickaxe with Q while the hotbar was locked.
        action = noop(); action['drop'] = 1; action['inventory'] = 1; action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'iron_ore': 1 if n >= 3 else 0}, selected='stone_pickaxe'))
        report = self.session(env, FakePolicy(action), tool_lock=True).run(
            [Step('s1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual([(a['drop'], a['inventory'], a['attack']) for a in env.actions[1:]], [(0, 0, 1)] * 2)
        telemetry = [json.loads(l) for l in (self.out / 'run/telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([t.get('masked') for t in telemetry if t['controller'] == 'policy'], [['drop', 'inventory']] * 2)
        self.assertEqual(report['stage_results'][0]['tool_masked'], 2)

    def test_escape_climb_ends_on_the_surface_not_above_it(self):
        # Dev check D1 (2026-11-05): the first segment already reached the surface; the second still required +3 and the
        # policy built a tower two blocks above the ground, where the craft found no placement surface.
        climb = lambda: Step('climb-2', 'dig', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise', 'to_surface': True})
        env = FakeEnv(lambda n, a: info({'oak_planks': 3}, selected='oak_planks', y=71., demo_fixture_origin_y=71))
        report = self.session(env, fixture=0, limit_s=5., clock=Clock(.5)).run([climb()])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage.get('completion'), stage['steps']), ('completed', 'already_at_surface', 0))
        ys = {1: 66., 2: 66., 3: 67.25, 4: 68., 5: 68., 6: 69.2, 7: 69., 8: 69.}
        env = FakeEnv(lambda n, a: info({'oak_planks': 3}, selected='oak_planks', y=ys.get(n, 69.), demo_fixture_origin_y=69))
        self.recorder.close(); self.gate = StopGate(); self.out = self.out / 'second'  # a second, independent session
        report = self.session(env, fixture=0, limit_s=8., clock=Clock(.25)).run([climb()])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage.get('completion'), env.calls), ('completed', 'server_surface', 8))

    def test_policy_step_after_a_rule_step_starts_with_a_level_view(self):
        # Evaluation round 2, E01: the iron step began looking down at the table the craft had just used and the policy
        # kept digging the floor for 13 minutes. A policy step after a rule step first levels the view (disclosed).
        env = FakeEnv(lambda n, a: info({'wooden_pickaxe': 1, 'iron_ore': 1 if n >= 12 else 0}, selected='wooden_pickaxe',
                                        table=(1, 2, 3), pitch=55.))
        steps = [Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1, params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'}),
                 Step('iron', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        self.session(env).run(steps)
        levels = [l for l in self.timeline() if l['kind'] == 'level_view']
        self.assertEqual((len(levels), levels[0]['step_id'], levels[0]['pitch_before']), (1, 'iron', 55.))
        self.assertIn([-55., 0.], [list(a['camera']) for a in env.actions])
        env2 = FakeEnv(lambda n, a: info({'iron_ore': 1 if n >= 3 else 0}, pitch=55.))
        self.recorder.close(); self.gate = StopGate(); self.out = self.out / 'second'
        self.session(env2).run([Step('iron', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual([l for l in self.timeline() if l['kind'] == 'level_view'], [])  # first step: the policy's own view

    def test_tool_lock_masks_use_on_a_workstation(self):
        # Real check P3 (2026-11-05): right after the stone pickaxe craft the policy opened the crafting table (use) and,
        # with the inventory key locked, stayed in its GUI for 420 steps clicking the wooden pickaxe into the grid.
        action = noop(); action['use'] = 1; action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'iron_ore': 1 if n >= 3 else 0}, selected='stone_pickaxe', hit='crafting_table'))
        self.session(env, FakePolicy(action), tool_lock=True).run([Step('s1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual([(a['use'], a['attack']) for a in env.actions[1:]], [(0, 1)] * 2)
        telemetry = [json.loads(l) for l in (self.out / 'run/telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([t.get('masked') for t in telemetry if t['controller'] == 'policy'], [['use']] * 2)

    def test_use_on_other_blocks_passes_the_lock(self):
        action = noop(); action['use'] = 1
        env = FakeEnv(lambda n, a: info({'dirt': 3}, selected='dirt', hit='stone', y=64. + (n >= 3)))
        self.session(env, FakePolicy(action), tool_lock=True).run([Step('s1', 'dig', 'policy', 'rise', 1, params={'until': 'rise'})])
        self.assertEqual(env.actions[1]['use'], 1)  # pillar placement is the climb itself

    def test_open_gui_during_a_policy_step_is_closed_by_the_session(self):
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'iron_ore': 1 if n >= 6 else 0}, selected='stone_pickaxe', screen=2 if 2 <= n <= 3 else 0))
        report = self.session(env, FakePolicy(action), tool_lock=True).run([Step('s1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')])
        self.assertEqual(report['stage_results'][0]['status'], 'completed')
        telemetry = [json.loads(l) for l in (self.out / 'run/telemetry.jsonl').read_text().splitlines()]
        presses = [t for t in telemetry if t['controller'] == 'gui_close' and t['action'].get('inventory') == 1]
        self.assertTrue(presses); self.assertEqual(report['stage_results'][0]['gui_closed'], len(presses))
        self.assertTrue(all(a['inventory'] == 0 for a, b in zip(env.actions, env.actions[1:]) if b['inventory']))  # press, release

    def test_tool_lock_count_survives_time_limit(self):
        action = noop(); action['hotbar.2'] = 1
        env = FakeEnv(lambda n, a: info())
        report = self.session(env, FakePolicy(action), limit_s=2., clock=Clock(.5), tool_lock=True).run(
            [Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'time_limit')
        self.assertEqual(stage['tool_masked'], len(env.actions) - 1)

    def test_tool_keys_pass_without_lock(self):
        action = noop(); action['hotbar.3'] = 1; action['drop'] = 1
        env = FakeEnv(lambda n, a: info({'cobblestone': 3 if n >= 3 else 0}))
        report = self.session(env, FakePolicy(action)).run([Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual([(a['hotbar.3'], a['drop']) for a in env.actions[1:]], [(1, 1), (1, 1)])
        self.assertNotIn('tool_masked', report['stage_results'][0])
        self.assertFalse(report['tool_lock'])

    def test_probe_until_rise_completes_on_server_height_gain(self):
        # Pit-escape probe: success is the server feet height rising by `count` blocks from the step start.
        env = FakeEnv(lambda n, a: info(y=60. + (6 if n >= 5 else n - 1)))
        report = self.session(env).run([Step('up', 'dig', 'policy', 'height', 6, params={'until': 'rise'})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['completion']), ('completed', 'server_rise'))
        self.assertEqual(stage['rise'], 6.)
        self.assertEqual(env.calls, 6)  # the height counts once the player stands (two equal samples)

    def test_rise_counts_standing_heights_not_a_jump_apex(self):
        # Real check P4 (2026-11-05): climb segments "finished" at a jump apex (70.0 while standing on 69), so the craft
        # after the escape failed two blocks down.
        # Real check P5: the server reports the apex height twice (68.24918707874468), so standing also needs a whole-block height.
        apex = 68.24918707874468
        ys = {1: 65., 2: 66.25, 3: 67., 4: 68.0013, 5: apex, 6: apex, 7: 67.5, 8: 67., 9: 67., 10: 68., 11: 68.}
        env = FakeEnv(lambda n, a: info({'dirt': 3}, selected='dirt', y=ys.get(n, 68.)))
        report = self.session(env).run([Step('climb-1', 'dig', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['rise'], env.calls), ('completed', 3., 11))

    def test_probe_rise_is_reported_on_time_limit(self):
        env = FakeEnv(lambda n, a: info(y=60. if n == 1 else 62.))
        report = self.session(env, limit_s=3., clock=Clock(.5)).run(
            [Step('up', 'dig', 'policy', 'height', 6, params={'until': 'rise'})])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'time_limit')
        self.assertEqual(stage['rise'], 2.)

    def ceiling_script(self, breaks):
        # Start block (5, 64, 9); breaks are server destroy positions in order.
        def script(n, a):
            k = sum(1 for at in breaks if n >= at[0])
            x, y, z = breaks[k - 1][1] if k else (-999, -999, -999)
            return info(x=5.5, y=64., z=9.5, destroys=k, demo_last_destroy_x=x, demo_last_destroy_y=y, demo_last_destroy_z=z)
        return script

    def test_probe_until_clear_above_counts_breaks_over_the_head(self):
        # Ceiling probe: only blocks broken in the start column above head height count (a wall break does not).
        env = FakeEnv(self.ceiling_script([(3, (5, 66, 9)), (5, (6, 66, 9)), (7, (5, 67, 9))]))
        report = self.session(env).run([Step('clear', 'dig', 'policy', 'ceiling', 2, params={'until': 'clear_above'})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['completion'], stage['cleared']), ('completed', 'server_clear_above', 2))
        self.assertEqual(env.calls, 7)

    def test_probe_clear_above_is_reported_on_time_limit(self):
        env = FakeEnv(self.ceiling_script([(3, (5, 66, 9))]))
        report = self.session(env, limit_s=3., clock=Clock(.5)).run(
            [Step('clear', 'dig', 'policy', 'ceiling', 2, params={'until': 'clear_above'})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['cleared']), ('time_limit', 1))

    def oversee(self, goal, steps, claims=(), llm=None):
        from minecraftrl.pipeline.kb import Evidence, KnowledgeBase
        from minecraftrl.pipeline.oversight import Oversight
        self.kb = KnowledgeBase(self.out / 'kb.sqlite')
        for subject, relation, obj in claims:
            self.kb.add_claim(subject, relation, obj, {}, [Evidence('web', 'support', 'https://minecraft.wiki')], run_id='seed')
        self.addCleanup(self.kb.close)
        return Oversight(self.kb, {'game_version': '1.16.5'}, goal, steps, run_id='test', llm=llm,
                         selected={'log': 'wood', 'cobblestone': 'dig', 'iron_ore': 'dig', 'escape': 'climb'})

    def llm(self, claims=None, fail=False):
        from test_pipeline_knowledge import FakeTransport, candidate, response
        from minecraftrl.pipeline.openai_client import ResponsesClient
        claims = claims if claims is not None else [
            candidate(), candidate(relation='drops', obj='iron_ore', conditions={'game_version': '1.16.5'})]
        self.transport = FakeTransport(response(claims), fail=fail)
        return ResponsesClient('gpt-test', 'sk-test', allow_paid=True, transport=self.transport, max_calls=2)

    def test_hold_acquires_knowledge_then_resumes_with_new_plan(self):
        # Guard holds on unknown iron; the LLM supplies cited knowledge; the planner (stone pickaxe owned) replans; the
        # resumed step mines iron with the tool and the run completes on server inventory.
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'stone_pickaxe': 1, 'iron_ore': 1 if n >= 6 else 0}, hit='iron_ore', tick=1000 + n,
                                        selected='stone_pickaxe'))
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore', params={'precondition': 'unverified'})]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps, llm=self.llm())).run(steps)
        self.assertEqual(report['status'], 'completed'); self.assertTrue(report['all_steps_completed'])
        held, resumed = report['stage_results']
        self.assertEqual((held['status'], held['slow_path']['route'], held['slow_path']['resolution']), ('anomaly_hold', 'acquire_then_replan', 'replan'))
        self.assertEqual(len(held['slow_path']['claim_ids']), 2); self.assertEqual(held['slow_path']['llm'], 'called')
        self.assertEqual((resumed['status'], resumed['skill']), ('completed', 'dig'))
        self.assertGreater(sum(a['attack'] for a in env.actions), 0)  # attacks resume only with a known-sufficient tool
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(report['oversight']['api_calls'], 1)  # summarised at the end, not at the start (A4 said 0)
        self.assertEqual([c['ok'] for c in report['oversight']['call_log']], [True])  # diagnosis of failed calls (round 1)
        kinds = [line['kind'] for line in self.timeline()]
        for kind in ('llm_request', 'kb_change', 'plan', 'resume'): self.assertIn(kind, kinds)

    def test_missing_knowledge_is_asked_once_more(self):
        from test_pipeline_knowledge import FakeTransport, candidate, response
        from minecraftrl.pipeline.openai_client import ResponsesClient
        first = response([candidate(), candidate(relation='drops', obj='iron_ore')])  # no recipe for the pickaxe yet
        second = response([candidate(subject='stone_pickaxe', relation='recipe', obj={'ingredients': {'cobblestone': 3, 'stick': 2},
                                                                                      'count': 1, 'grid': 3})])
        replies = [first, second]
        transport = FakeTransport(); transport.reply = None
        def reply(request, timeout): transport.requests.append(json.loads(request.data)); return replies[len(transport.requests) - 1]
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=reply, max_calls=2)
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps, llm=llm)).run(steps)
        slow = report['stage_results'][0]['slow_path']
        self.assertEqual(len(transport.requests), 2)
        self.assertIn('stone_pickaxe', json.dumps(transport.requests[1]['input']))
        self.assertIn('which block drops it', json.dumps(transport.requests[1]['input']))  # round 3c: ask for the source block
        self.assertEqual(slow['rounds'], 2)

    def test_a_chain_missing_link_by_link_is_asked_round_by_round(self):
        # Round 3b E03 (2026-11-05): the first answer lost its recipes as uncited; each later round only reveals the next
        # missing link (stone pickaxe -> wooden pickaxe -> planks/sticks), so two rounds per hold could not finish the chain.
        from test_pipeline_knowledge import FakeTransport, candidate, response
        from minecraftrl.pipeline.openai_client import ResponsesClient
        R = lambda subject, ingredients, grid, count=1: candidate(subject=subject, relation='recipe',
                                                                   obj={'ingredients': ingredients, 'count': count, 'grid': grid})
        replies = [response([candidate(), candidate(relation='drops', obj='iron_ore'),
                             candidate(subject='stone', obj={'tool_class': 'pickaxe', 'min_tier': 'wooden'}),
                             candidate(subject='stone', relation='drops', obj='cobblestone'),
                             candidate(subject='log', obj={'tool_class': None, 'min_tier': 'hand'})]),
                   response([R('stone_pickaxe', {'cobblestone': 3, 'stick': 2}, 3)]),
                   response([R('wooden_pickaxe', {'planks': 3, 'stick': 2}, 3)]),
                   response([R('planks', {'log': 1}, 2, 4), R('stick', {'planks': 2}, 2, 4), R('crafting_table', {'planks': 4}, 2)])]
        transport = FakeTransport()
        def reply(request, timeout): transport.requests.append(json.loads(request.data)); return replies[len(transport.requests) - 1]
        llm = ResponsesClient('gpt-test', 'k', allow_paid=True, transport=reply, max_calls=8)
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps, llm=llm)).run(steps)
        slow = report['stage_results'][0]['slow_path']
        self.assertEqual((slow['rounds'], slow['resolution']), (4, 'replan'))
        self.assertEqual([s['skill'] for s in slow['plan']][-1], 'dig')

    def test_persistent_violation_retries_the_step_then_stops(self):
        # Real run A2 (2026-11-04): a KB-hit replan of the same plan re-held instantly 232 times; the violation timer
        # must restart with the step and retries must be bounded.
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('logs', 'dig', 'policy', 'log', 3, block='log')]
        oversight = self.oversee('iron_ore', steps, [('iron_ore', 'harvest_requires', {'tool_class': 'pickaxe', 'min_tier': 'stone'})])
        oversight.guard.violation_hold_s = 3.; oversight.max_retries = 2
        report = self.session_with(env, FakePolicy(action), oversight, limit_s=900., clock=Clock(.25)).run(steps)
        stages = report['stage_results']
        self.assertEqual([s['slow_path']['resolution'] for s in stages], ['retry', 'retry', 'stop'])
        self.assertTrue(all(s['anomaly']['kind'] == 'guard_violation_persistent' for s in stages))
        self.assertTrue(all(s['steps'] > 0 for s in stages))  # every retry dispatched masked actions first
        self.assertEqual(sum(a['attack'] for a in env.actions), 0)
        self.assertEqual(report['status'], 'anomaly_hold')
        masks = [l for l in self.timeline() if l['kind'] == 'monitor_event' and l['event']['kind'] == 'guard_violation']
        self.assertEqual(len(masks), 3)  # first mask per step attempt is recorded

    def test_retry_after_a_persistent_violation_turns_away_first(self):
        # Evaluation round 1, E01 (2026-11-05): the wood policy kept hitting the stone wall it faced; each retry restarted
        # from the same view. The retry now turns 180 degrees and levels the view first (disclosed rule action).
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n, pitch=40.))
        steps = [Step('logs', 'dig', 'policy', 'log', 3, block='log')]
        oversight = self.oversee('iron_ore', steps, [('iron_ore', 'harvest_requires', {'tool_class': 'pickaxe', 'min_tier': 'stone'})])
        oversight.guard.violation_hold_s = 3.; oversight.max_retries = 1
        report = self.session_with(env, FakePolicy(action), oversight, limit_s=900., clock=Clock(.25)).run(steps)
        turns = [a for a in env.actions if abs(a['camera'][1]) == 180.]
        self.assertEqual(len(turns), 1); self.assertEqual(turns[0]['camera'][0], -40.)
        self.assertEqual([l['kind'] for l in self.timeline()].count('turn_away'), 1)
        self.assertEqual(report['stage_results'][-1]['slow_path']['resolution'], 'stop')

    def test_llm_failure_stops_at_the_hold(self):
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps, llm=self.llm(fail=True))).run(steps)
        slow = report['stage_results'][0]['slow_path']
        self.assertEqual((report['status'], slow['llm'], slow['resolution']), ('anomaly_hold', 'failed', 'stop'))
        self.assertEqual(sum(a['attack'] for a in env.actions), 0)

    def test_uncited_knowledge_is_not_used_for_replanning(self):
        from test_pipeline_knowledge import candidate
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        llm = self.llm([candidate(urls=('https://not-cited.example/x',))])
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps, llm=llm)).run(steps)
        slow = report['stage_results'][0]['slow_path']
        # Nothing usable after the second (missing-only) round either: the hold stays and the run stops.
        self.assertEqual((slow['claim_ids'], slow['rejected'], slow['rounds'], slow['resolution']), ([], 2, 2, 'stop'))

    def session_with(self, env, policy, oversight, **kw):
        self.recorder = Recorder(self.out / 'run')
        return DemoSession(env, policy, None, self.recorder, self.gate, self.stop, limit_s=kw.get('limit_s', 60.), skills=SKILLS,
                           clock=kw.get('clock', Clock()), rule_controller=kw.get('controller', FakeController), settle_steps=0,
                           oversight=oversight, fixture_sha52=kw.get('fixture'))

    def pit(self, claims=(), llm=None):
        """Real flow state (run B, 2026-11-04): 6 blocks below the surface after the cobblestone step; the stone pickaxe
        craft cannot place its table there."""
        from test_pipeline_planner import iron_kb
        class PitController(FakeController):
            def actions(self):
                if self.goals != ['stone_pickaxe']: yield from super().actions()
                yield self.environment.env.raw.action_space.noop()
                raise ValueError('no reachable safe placement surface')
        inventory = {'dirt': 3, 'cobblestone': 3, 'oak_planks': 3, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stick': 2}
        env = FakeEnv(lambda n, a: info(inventory, tick=1000 + n, y=65. if n == 1 else 59.))
        steps = [Step('craft-stone_pickaxe', 'craft', 'rule', 'stone_pickaxe', 1,
                      params={'goals': ['stone_pickaxe'], 'equip': 'stone_pickaxe'})]
        oversight = self.oversee('iron_ore', steps, claims, llm=llm); iron_kb(self.kb)
        report = self.session_with(env, FakePolicy(), oversight, controller=PitController).run(steps)
        return report, report['stage_results'][0]

    def test_craft_blocked_in_the_pit_inserts_an_escape_from_sqlite(self):
        report, held = self.pit([('vertical_pit', 'escape_by', {'method': 'pillar_up', 'blocks_per_level': 1})])
        self.assertEqual((held['status'], held['anomaly']['kind'], held['anomaly']['hold']), ('anomaly_hold', 'rule_skill_failed', True))
        self.assertEqual(held['anomaly']['evidence']['depth'], 6.)
        self.assertIn('no reachable safe placement surface', held['anomaly']['evidence']['reason'])
        slow = held['slow_path']
        self.assertEqual((slow['subject'], slow['relation'], slow['route'], slow['llm'], slow['resolution']),
                         ('vertical_pit', 'escape_by', 'replan_from_kb', 'not_needed', 'replan'))
        self.assertEqual([(s['skill'], s['target']) for s in slow['plan']], [
            ('equip', 'dirt'), ('climb', 'rise'), ('equip', 'oak_planks'), ('climb', 'rise'), ('craft', 'stone_pickaxe'),
            ('dig', 'iron_ore')])
        resumed = report['stage_results'][1]
        self.assertEqual((resumed['id'], resumed['kind']), ('equip-dirt', 'rule'))

    def test_pit_depth_uses_the_fixture_origin_when_the_player_starts_in_the_shaft(self):
        # Real check P1 (2026-11-05): the shaft is dug under the start block, so the first observation is already at its
        # floor (depth 0 was recorded); the surface is the fixture origin (the start block), reported by the game.
        from test_pipeline_planner import iron_kb
        inventory = {'dirt': 3, 'oak_planks': 3, 'cobblestone': 3, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stick': 2}
        env = FakeEnv(lambda n, a: info(inventory, tick=1000 + n, y=65., demo_fixture_origin_y=71))
        steps = [Step('craft-stone_pickaxe', 'craft', 'rule', 'stone_pickaxe', 1, params={'goals': ['stone_pickaxe'], 'equip': 'stone_pickaxe'})]
        oversight = self.oversee('iron_ore', steps, [('vertical_pit', 'escape_by', {'method': 'pillar_up', 'blocks_per_level': 1})])
        iron_kb(self.kb)
        report = self.session_with(env, FakePolicy(), oversight, controller=FailingController, fixture=0).run(steps)
        evidence = report['stage_results'][0]['anomaly']['evidence']
        self.assertEqual((evidence['surface_y'], evidence['depth']), (71., 6.))

    def test_pit_escape_knowledge_is_acquired_when_sqlite_has_none(self):
        from test_pipeline_knowledge import candidate
        llm = self.llm([candidate(subject='vertical_pit', relation='escape_by', obj={'method': 'pillar_up', 'blocks_per_level': 1})])
        _, held = self.pit(llm=llm)
        slow = held['slow_path']
        self.assertEqual((slow['route'], slow['llm'], slow['rounds'], slow['resolution']), ('acquire_then_replan', 'called', 1, 'replan'))
        request = json.loads(self.transport.requests[0]['input'][1]['content'])
        self.assertEqual((request['subject'], request['relation'], request['situation']['depth']), ('vertical_pit', 'escape_by', 6.))

    def test_climb_that_runs_out_of_blocks_replans_from_the_current_height(self):
        # A segment's stack ran out with levels to go (e.g. a block placed sideways): the escape is planned again from
        # where the player now stands, with the stacks that are left (SQLite hit, no LLM).
        from test_pipeline_planner import iron_kb
        inventory = {'dirt': 2, 'oak_planks': 3, 'cobblestone': 3, 'crafting_table': 1, 'wooden_pickaxe': 1, 'stick': 2}
        env = FakeEnv(lambda n, a: info(inventory, tick=1000 + n, y=65. if n == 1 else 59.))
        steps = [Step('climb-1', 'dig', 'policy', 'rise', 3, params={'until': 'rise', 'expect': 'rise'})]
        oversight = self.oversee('iron_ore', steps, [('vertical_pit', 'escape_by', {'method': 'pillar_up', 'blocks_per_level': 1})])
        iron_kb(self.kb)
        report = self.session_with(env, FakePolicy(), oversight).run(steps)
        held = report['stage_results'][0]
        self.assertEqual((held['anomaly']['kind'], held['anomaly']['evidence']['depth']), ('climb_without_block', 6.))
        slow = held['slow_path']
        self.assertEqual((slow['relation'], slow['route'], slow['llm'], slow['resolution']), ('escape_by', 'replan_from_kb', 'not_needed', 'replan'))
        self.assertEqual([(s['skill'], s['target'], s['count']) for s in slow['plan']][:4], [
            ('equip', 'oak_planks', 1), ('climb', 'rise', 3), ('equip', 'dirt', 1), ('climb', 'rise', 3)])

    def iron_done(self, llm=None):
        """Iron mined with a stone pickaxe: the server shows iron_ore gained, harvestable. The KB holds the real gpt-5.4 claim
        (A3/A4) that 1.16.5 iron ore drops raw_iron."""
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'stone_pickaxe': 1, 'iron_ore': 1 if n >= 4 else 0}, hit='iron_ore', tick=1000 + n,
                                        selected='stone_pickaxe', destroys=1 if n >= 3 else 0, demo_last_destroyed_block='iron_ore',
                                        demo_last_destroy_harvestable=1, demo_last_destroy_x=0, demo_last_destroy_y=65, demo_last_destroy_z=3))
        oversight = self.oversee('iron_ore', [], [('iron_ore', 'harvest_requires', {'tool_class': 'pickaxe', 'min_tier': 'stone'})], llm=llm)
        from minecraftrl.pipeline.kb import Evidence
        raw = self.kb.add_claim('iron_ore', 'drops', 'raw_iron', {'game_version': '1.16.5'},
                                [Evidence('web', 'support', 'https://minecraft.wiki/w/Iron_Ore'), Evidence('llm_assertion', 'support', 'model:gpt')],
                                run_id='seed')
        need, = self.kb.query('iron_ore', 'harvest_requires', context={'game_version': '1.16.5'})
        steps = [Step('mine-iron_ore', 'dig', 'policy', 'iron_ore', 1, block='iron_ore', claim_ids=(need.id,))]
        report = self.session_with(env, FakePolicy(action), oversight).run(steps)
        return report, raw, need.id

    def test_timeline_carries_kb_snapshots_for_the_replay(self):
        report, raw, _ = self.iron_done()
        snaps = [l for l in self.timeline() if l['kind'] == 'kb_snapshot']
        self.assertGreaterEqual(len(snaps), 2)
        first, last = snaps[0], snaps[-1]
        self.assertEqual(first['reason'], 'session_start'); self.assertEqual(last['reason'], 'step_verified')
        row, = [c for c in last['claims'] if c['id'] == raw]
        self.assertEqual((row['status'], row['sources'], row['observed']), ('disputed', ['https://minecraft.wiki/w/Iron_Ore'], {'support': 0, 'refute': 1}))

    def test_lossy_break_refutes_the_permitting_knowledge_then_asks_again(self):
        # Evaluation round 2, E03 (spec 2 post-hoc scene): the store said stone needs only the hand; bare hands broke stone
        # with no drop. The permitting claim gets a game refutation (disputed) and the LLM is asked again with the observation.
        from minecraftrl.pipeline.kb import Evidence
        from test_pipeline_knowledge import candidate
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info({}, hit='stone', tick=1000 + n, selected='air', destroys=1 if n >= 3 else 0,
                                        demo_last_destroyed_block='stone', demo_last_destroy_harvestable=0,
                                        demo_last_destroy_x=0, demo_last_destroy_y=63, demo_last_destroy_z=1))
        steps = [Step('mine-cobblestone', 'dig', 'policy', 'cobblestone', 3, block='stone')]
        llm = self.llm([candidate(subject='stone', obj={'tool_class': 'pickaxe', 'min_tier': 'wooden'})])
        oversight = self.oversee('iron_ore', steps, llm=llm)
        wrong = self.kb.add_claim('stone', 'harvest_requires', {'tool_class': 'pickaxe', 'min_tier': 'hand'}, {'game_version': '1.16.5'},
                                  [Evidence('web', 'support', 'https://minecraft.wiki/w/Stone')], run_id='seed')
        report = self.session_with(env, FakePolicy(action), oversight).run(steps)
        held = report['stage_results'][0]
        self.assertEqual(held['anomaly']['kind'], 'destroy_not_harvestable')
        claim, = [c for c in self.kb.query('stone', 'harvest_requires', context={'game_version': '1.16.5'}, usable_only=False) if c.id == wrong]
        self.assertEqual((claim.status, claim.refute), ('disputed', 1))
        conflict, = [l for l in self.timeline() if l['kind'] == 'knowledge_conflict']
        self.assertEqual((conflict['claim_id'], conflict['relation'], conflict['outcome']), (wrong, 'harvest_requires', 'disputed'))
        slow = held['slow_path']
        self.assertEqual((slow['route'], slow['llm']), ('acquire_then_replan', 'called'))
        query = json.loads(self.transport.requests[0]['input'][1]['content'])
        self.assertEqual((query['observation']['broke'], query['observation']['harvestable']), ('stone', False))
        usable = self.kb.query('stone', 'harvest_requires', context={'game_version': '1.16.5'})
        self.assertEqual([c.object['min_tier'] for c in usable], ['wooden'])

    def test_used_knowledge_gains_game_support_when_a_step_succeeds(self):
        report, _, need = self.iron_done()
        self.assertEqual(report['status'], 'completed')
        claim, = [c for c in self.kb.query('iron_ore', 'harvest_requires', context={'game_version': '1.16.5'}) if c.id == need]
        self.assertEqual(claim.status, 'supported')

    def test_contradicted_drop_is_split_by_a_cited_version_condition(self):
        # Spec 2: the wiki/LLM "iron ore -> raw iron" meets 1.16.5 reality; the claim is not deleted but narrowed to >= 1.17.
        from test_pipeline_knowledge import FakeTransport
        from minecraftrl.pipeline.openai_client import ResponsesClient
        wiki = 'https://minecraft.wiki/w/Raw_Iron'
        reply = {'status': 'completed', 'usage': {}, 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'annotations': [
            {'type': 'url_citation', 'url': wiki}], 'text': json.dumps({'condition_json': '{"game_version": {"gte": "1.17"}}',
                                                                          'source_urls': [wiki], 'excerpt': 'Raw iron was added in 1.17.'})}]}]}
        self.transport = FakeTransport(reply)
        report, raw, _ = self.iron_done(ResponsesClient('gpt-test', 'k', allow_paid=True, transport=self.transport, max_calls=2))
        self.assertEqual(report['status'], 'completed')  # the goal criterion is the server inventory, not the knowledge
        old = self.kb.query('iron_ore', 'drops', context={'game_version': '1.16.5'}, usable_only=False)
        self.assertEqual([c.status for c in old if c.id == raw], ['retracted'])
        newer, = self.kb.query('iron_ore', 'drops', context={'game_version': '1.17.1'})
        self.assertEqual((newer.object, newer.conditions, newer.status), ('raw_iron', {'game_version': {'gte': '1.17'}}, 'proposed'))
        conflict, = [l for l in self.timeline() if l['kind'] == 'knowledge_conflict']
        self.assertEqual((conflict['claim_id'], conflict['outcome'], conflict['observed']), (raw, 'conditionalized', {'iron_ore': 1}))
        self.assertEqual(report['oversight']['api_calls'], 1)

    def test_contradicted_drop_without_llm_is_disputed(self):
        report, raw, _ = self.iron_done()
        claim, = [c for c in self.kb.query('iron_ore', 'drops', context={'game_version': '1.16.5'}, usable_only=False) if c.id == raw]
        self.assertEqual(claim.status, 'disputed')

    def ceiling(self, claims=(), llm=None):
        """Prepared ceiling scenario (user decision 2026-11-04): two dirt blocks over the head; the climb jumps without rising
        until the clear step opens the view up, then rises."""
        state = dict(attacks=0, up=False, jumps=0)
        def script(n, a):
            state['attacks'] += bool(a.get('attack')); state['up'] |= a['camera'][0] < -80
            cleared = min(2, state['attacks'] // 3)
            if cleared == 2 and a.get('jump'): state['jumps'] += 1
            return info({'dirt': 16}, tick=1000 + n, selected='dirt', pitch=-90. if state['up'] else 0., hit='dirt' if cleared < 2 else 'air',
                        destroys=cleared, y=64. + (min(6, state['jumps'] // 2) if cleared == 2 else (n % 2) * .2),
                        demo_last_destroyed_block='dirt', demo_last_destroy_harvestable=1, demo_last_destroy_x=0,
                        demo_last_destroy_y=66, demo_last_destroy_z=0)
        action = noop(); action['jump'] = 1
        steps = [Step('climb-1', 'dig', 'policy', 'rise', 6, params={'until': 'rise', 'expect': 'rise'})]
        oversight = self.oversee('iron_ore', steps, claims, llm=llm)
        report = self.session_with(FakeEnv(script), FakePolicy(action), oversight, limit_s=120.).run(steps)
        return report

    def test_ceiling_is_cleared_by_the_disclosed_rule_then_the_climb_resumes(self):
        report = self.ceiling([('solid_block', 'blocks_movement', True)])
        held, cleared, climbed = report['stage_results']
        self.assertEqual((held['anomaly']['kind'], held['slow_path']['route'], held['slow_path']['resolution']),
                         ('blocked_overhead', 'replan_from_kb', 'insert'))
        self.assertEqual([(s['skill'], s['kind']) for s in held['slow_path']['plan']], [('clear_above', 'rule'), ('dig', 'policy')])
        self.assertEqual((cleared['skill'], cleared['status'], cleared['cleared'], cleared['controller']), ('clear_above', 'completed', 2, 'rule'))
        self.assertEqual((climbed['status'], climbed['completion']), ('completed', 'server_rise'))
        self.assertEqual(report['status'], 'completed')
        request, = [l for l in self.timeline() if l['kind'] == 'learning_request']
        self.assertEqual((request['rule_skill'], request['status']), ('clear_above', 'recorded_not_executed'))
        self.assertIn('mine the ceiling', request['policy_candidates'])

    def lagged_ceiling(self, blocks):
        """Real C1/C2 timing: the crosshair shows air 1 tick after a block breaks (2 ticks of air before the next block shows),
        and the server destroy count arrives 4 ticks after that."""
        state = dict(left=blocks, attacks=0, broken=[])
        def script(n, a):
            if a.get('attack') and state['left']:
                state['attacks'] += 1
                if state['attacks'] == 3: state['attacks'] = 0; state['left'] -= 1; state['broken'].append(n)
            destroys = sum(n >= b + 4 for b in state['broken'])
            lag = state['broken'] and n - state['broken'][-1] < 3
            hit = 'air' if not state['left'] or lag else 'dirt'
            return info({'dirt': 16}, tick=1000 + n, selected='dirt', pitch=-90. if n > 8 else 0., hit=hit, destroys=destroys)
        return script

    def test_clear_above_waits_for_the_server_before_judging_the_view(self):
        env = FakeEnv(self.lagged_ceiling(2))
        report = self.session(env).run([Step('clear-above', 'clear_above', 'rule', 'cleared', 1, params={'goals': ['cleared']})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['cleared'], stage['completion']), ('completed', 2, 'view_up_open'))

    def test_clear_above_with_an_open_view_is_already_done(self):
        env = FakeEnv(self.lagged_ceiling(0))
        report = self.session(env).run([Step('clear-above', 'clear_above', 'rule', 'cleared', 1, params={'goals': ['cleared']})])
        stage = report['stage_results'][0]
        self.assertEqual((stage['status'], stage['cleared']), ('completed', 0))

    def test_repeated_ceiling_holds_keep_one_climb_id_and_stay_bounded(self):
        # Real C1: each insert renamed the climb (-after-clear-after-clear), so the per-step retry limit never applied.
        action = noop(); action['jump'] = 1
        env = FakeEnv(lambda n, a: info({'dirt': 16}, tick=1000 + n, y=64. + (n % 2) * .2, selected='dirt', pitch=-90., hit='air'))
        steps = [Step('climb-1', 'dig', 'policy', 'rise', 6, params={'until': 'rise', 'expect': 'rise'})]
        oversight = self.oversee('iron_ore', steps, [('solid_block', 'blocks_movement', True)]); oversight.max_retries = 2
        report = self.session_with(env, FakePolicy(action), oversight, limit_s=600.).run(steps)
        climbs = [s['id'] for s in report['stage_results'] if s['skill'] == 'dig']
        self.assertEqual(climbs, ['climb-1'] + ['climb-1-after-clear'] * 3)  # holds 2-3 resume, the 4th exceeds the limit
        self.assertEqual(report['stage_results'][-1]['slow_path']['resolution'], 'stop')

    def test_ceiling_knowledge_is_acquired_when_sqlite_has_none(self):
        from test_pipeline_knowledge import candidate
        llm = self.llm([candidate(subject='solid_block', relation='blocks_movement', obj=True)])
        report = self.ceiling(llm=llm)
        slow = report['stage_results'][0]['slow_path']
        self.assertEqual((slow['route'], slow['llm'], slow['resolution']), ('acquire_then_replan', 'called', 'insert'))
        self.assertEqual(report['status'], 'completed')

    def test_rule_failure_outside_a_pit_retries_the_step_by_default(self):
        # Spec 5.3 default when no decision exists: retry the same step, bounded; then the run fails at the hold.
        env = FakeEnv(lambda n, a: info(tick=1000 + n))
        steps = [Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1, params={'goals': ['wooden_pickaxe'], 'equip': None})]
        oversight = self.oversee('iron_ore', steps); oversight.max_retries = 2
        report = self.session_with(env, FakePolicy(), oversight, controller=FailingController).run(steps)
        slows = [s['slow_path'] for s in report['stage_results']]
        self.assertEqual([(s['relation'], s['route'], s['resolution']) for s in slows],
                         [(None, 'retry_stage', 'retry'), (None, 'retry_stage', 'retry'), (None, 'retry_limit', 'stop')])
        self.assertEqual(slows[0]['llm'], 'not_called'); self.assertEqual(report['status'], 'anomaly_hold')

    def test_rule_step_starts_after_pickups_settle(self):
        # Real run A4 (2026-11-05): a dirt drop was picked up one step after the craft snapshot (8 -> 9), so the server
        # inventory never matched and the recipe commit timed out. A rule step starts from a settled, closed inventory.
        FakeController.created.clear()
        env = FakeEnv(lambda n, a: info({'dirt': 8 if n <= 3 else 9, 'wooden_pickaxe': 1}, selected='wooden_pickaxe',
                                        screen=1 if n == 2 else 0, table=(1, 2, 3)))
        report = self.session(env).run([Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1,
                                             params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'})])
        self.assertEqual(report['stage_results'][0]['status'], 'completed')
        self.assertEqual(FakeController.created[0].inventory['dirt'], 9)
        self.assertEqual(sum(a['inventory'] for a in env.actions[:4]), 1)  # the open GUI was closed first

    def test_equip_only_rule_step_needs_no_workstation(self):
        env = FakeEnv(lambda n, a: info({'dirt': 3}, selected='dirt'))
        report = self.session(env).run([Step('equip-dirt', 'equip', 'rule', 'dirt', 1,
                                              params={'goals': ['dirt'], 'equip': 'dirt', 'equip_only': True})])
        self.assertEqual(report['stage_results'][0]['status'], 'completed')

    def test_guard_holds_before_an_unknown_iron_attack_is_sent(self):
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore', tick=1000 + n))
        steps = [Step('iron', 'dig', 'policy', 'iron_ore', 1, block='iron_ore')]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps)).run(steps)
        self.assertEqual(report['status'], 'anomaly_hold')
        self.assertEqual(sum(a['attack'] for a in env.actions), 0)  # the attack never reached the game
        stage = report['stage_results'][0]
        self.assertEqual((stage['anomaly']['kind'], stage['anomaly']['phase']), ('guard_unknown', 'pre_dispatch'))
        self.assertEqual((stage['slow_path']['hits'], stage['slow_path']['route'], stage['slow_path']['llm']),
                         ([], 'acquire_then_replan', 'not_called'))
        kinds = [line['kind'] for line in self.timeline()]
        self.assertLess(kinds.index('anomaly'), kinds.index('hold')); self.assertIn('kb_query', kinds)
        self.assertEqual(env.released, 1)

    def test_goal_without_iron_lets_the_same_hit_pass(self):
        action = noop(); action['attack'] = 1
        env = FakeEnv(lambda n, a: info({'oak_log': 3 if n >= 4 else 0}, hit='iron_ore', tick=1000 + n))
        steps = [Step('logs', 'dig', 'policy', 'log', 3, block='log')]
        oversight = self.oversee('log', steps, [('log', 'harvest_requires', {'tool_class': None, 'min_tier': 'hand'})])
        report = self.session_with(env, FakePolicy(action), oversight).run(steps)
        self.assertEqual(report['status'], 'completed'); self.assertNotIn('anomaly', report['stage_results'][0])

    def test_blocked_overhead_holds_the_climb_step(self):
        action = noop(); action['jump'] = 1
        env = FakeEnv(lambda n, a: info({'dirt': 16}, tick=1000 + n, y=64. + (n % 2) * .2, selected='dirt'))  # ceiling probe: dirt in hand
        steps = [Step('escape', 'dig', 'policy', 'height', 6, params={'expect': 'rise'})]
        report = self.session_with(env, FakePolicy(action), self.oversee('iron_ore', steps)).run(steps)
        stage = report['stage_results'][0]
        self.assertEqual((report['status'], stage['anomaly']['kind']), ('anomaly_hold', 'blocked_overhead'))
        self.assertEqual((stage['slow_path']['relation'], stage['slow_path']['route']), ('blocks_movement', 'acquire_then_replan'))
        self.assertLessEqual(env.calls, 25)

    def test_log_target_counts_any_log_kind(self):
        env = FakeEnv(lambda n, a: info({'oak_log': 1 if n >= 3 else 0, 'birch_log': 2 if n >= 4 else 0}))
        report = self.session(env).run([Step('s1', 'wood', 'policy', 'log', 3, block='log')])
        self.assertEqual(report['stage_results'][0]['status'], 'completed')

    def test_gain_is_relative_to_step_start(self):
        env = FakeEnv(lambda n, a: info({'cobblestone': 2 if n < 4 else 3}))
        report = self.session(env, limit_s=3., clock=Clock(.5)).run(
            [Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual(report['stage_results'][0]['status'], 'time_limit')

    def test_rule_step_drives_workshop_generator_until_goal(self):
        env = FakeEnv(lambda n, a: info({'wooden_pickaxe': 1}, selected='wooden_pickaxe', table=(1, 2, 3)))
        policy = FakePolicy()
        report = self.session(env, policy).run([Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1,
                                                     params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'})])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'completed'); self.assertEqual(stage['controller'], 'rule')
        self.assertEqual(policy.calls, 0)
        self.assertEqual(sum(a['use'] for a in env.actions), 3)
        samples = [json.loads(line) for line in (self.out / 'run/telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(sum(s['controller'] == 'rule' for s in samples), stage['steps'])
        uses = [i for i, s in enumerate(samples) if s['action'].get('use') == 1]
        self.assertEqual(uses, list(range(uses[0], uses[0] + 3)))  # the workshop's actions, after the inventory settles

    def test_rule_step_requires_server_equip(self):
        env = FakeEnv(lambda n, a: info({'wooden_pickaxe': 1}, selected='air', table=(1, 2, 3)))
        report = self.session(env, limit_s=4., clock=Clock(.5)).run(
            [Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1, params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'})])
        self.assertNotEqual(report['stage_results'][0]['status'], 'completed')

    def test_rule_step_exception_fails_stage_without_crash(self):
        env = FakeEnv(lambda n, a: info())
        report = self.session(env, controller=FailingController).run(
            [Step('c1', 'craft', 'rule', 'wooden_pickaxe', 1, params={'goals': ['wooden_pickaxe'], 'equip': None})])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'rule_skill_failed'); self.assertIn('open 2x2 inventory', stage['reason'])
        self.assertEqual(report['status'], 'step_failed'); self.assertEqual(env.released, 1)

    def test_limit_exceeded_marks_time_limit_not_success(self):
        env = FakeEnv(lambda n, a: info())
        report = self.session(env, limit_s=5., clock=Clock(1.)).run(
            [Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual(report['status'], 'time_limit')
        self.assertEqual(report['stage_results'][0]['status'], 'time_limit')
        self.assertFalse(report['all_steps_completed']); self.assertTrue(report['execution_valid'])

    def test_fixture_wait_requires_applied_state_and_hash(self):
        env = FakeEnv(lambda n, a: info({'cobblestone': 3 if n >= 6 else 0}, fixture_state=2 if n >= 3 else 1, sha52=77))
        policy = FakePolicy()
        report = self.session(env, policy, fixture=77).run([Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['prepare_steps'], 3); self.assertEqual(policy.calls, 3)

    def test_fixture_error_stops_before_policy(self):
        env = FakeEnv(lambda n, a: info(fixture_state=-1, sha52=0))
        policy = FakePolicy()
        report = self.session(env, policy, fixture=77).run([Step('s1', 'dig', 'policy', 'cobblestone', 3, block='stone')])
        self.assertEqual(report['status'], 'fixture_failed'); self.assertEqual(policy.calls, 0)
        self.assertFalse(report['execution_valid'])

    def test_probe_until_mining_start(self):
        env = FakeEnv(lambda n, a: info(mining='iron_ore' if n >= 4 else 'air'))
        report = self.session(env).run([Step('p1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore',
                                              params={'until': 'mining_start'})])
        stage = report['stage_results'][0]
        self.assertEqual(stage['status'], 'completed'); self.assertEqual(stage['completion'], 'server_mining')
        self.assertEqual(stage['gain'], 0)

    def test_probe_aim_with_attack_counts_before_dispatch(self):
        attack = noop(); attack['attack'] = 1
        env = FakeEnv(lambda n, a: info(hit='iron_ore' if n >= 2 else 'stone'))
        report = self.session(env, FakePolicy(attack)).run([Step('p1', 'dig', 'policy', 'iron_ore', 1, block='iron_ore',
                                                                 params={'until': 'mining_start'})])
        stage = report['stage_results'][0]
        self.assertEqual(stage['completion'], 'aim_attack'); self.assertEqual(stage['steps'], 1)

    def test_scripted_attack_records_destroy_outcome(self):
        def script(n, action):
            if n < 3: return info(mining='air', tick=1000 + n)
            if n < 8: return info(mining='iron_ore', tick=1000 + n)
            return info(mining='air', tick=1000 + n, destroys=1, demo_last_destroyed_block='iron_ore',
                        demo_last_destroy_harvestable=0, demo_last_destroy_tool='air', demo_last_destroy_tick=1008)
        env = FakeEnv(script)
        report = self.session(env).run([Step('t1', 'attack_hold', 'scripted', 'iron_ore', 1, block='iron_ore')])
        stage = report['stage_results'][0]
        self.assertEqual(stage['controller'], 'scripted_probe'); self.assertEqual(stage['status'], 'completed')
        self.assertEqual(stage['destroy'], dict(block='iron_ore', harvestable=0, tool='air', tick=1008,
                                                mining_start_tick=1003, mining_ticks=5))
        self.assertTrue(all(a['attack'] == 1 for a in env.actions[1:]))


class RecorderTests(unittest.TestCase):
    def test_recorder_lines_have_t_and_step(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = Recorder(Path(temp) / 'r', clock=lambda: 12.5)
            recorder.telemetry(5, {'tick': 1})
            recorder.event('guard', block='iron_ore')
            recorder.frame(5, np.zeros((4, 4, 3), np.uint8))
            recorder.close()
            line = json.loads((Path(temp) / 'r/timeline.jsonl').read_text())
            self.assertEqual(line, {'t': 12.5, 'step': 5, 'kind': 'guard', 'block': 'iron_ore'})
            sample = json.loads((Path(temp) / 'r/telemetry.jsonl').read_text())
            self.assertEqual(sample, {'t': 12.5, 'step': 5, 'tick': 1})
            self.assertTrue((Path(temp) / 'r/frames/000005.jpg').exists())


class LoaderTests(unittest.TestCase):
    def test_load_skills_checks_hash_shape_and_location(self):
        import hashlib
        from minecraftrl.pipeline.cli import load_skills
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'results/p').mkdir(parents=True)
            np.save(root / 'results/p/a.npy', np.zeros((1, 512), np.float32), allow_pickle=False)
            digest = hashlib.sha256((root / 'results/p/a.npy').read_bytes()).hexdigest()
            spec = dict(embedding='results/p/a.npy', sha256=digest, cond_scale=7.)
            skills = load_skills(root, {'a': spec, 'unused': dict(spec, sha256='0' * 64)}, ['a'])
            self.assertEqual(skills['a']['embedding'].shape, (1, 512)); self.assertNotIn('unused', skills)
            for bad in (dict(spec, sha256='0' * 64), dict(spec, embedding='../a.npy'), dict(spec, cond_scale=9.)):
                with self.subTest(bad=bad), self.assertRaises(ValueError): load_skills(root, {'a': bad}, ['a'])
            with self.assertRaises(KeyError): load_skills(root, {'a': spec}, ['missing'])

    def test_parse_separates_world_and_sampling_seed(self):
        from minecraftrl.pipeline.cli import parse, settings
        root = Path(__file__).resolve().parents[1]
        base = ['probe', '--root', str(root), '--config', str(root / 'configs/iron-demo.json'), '--plan-json', 'p.json']
        args = parse(base + ['--output', str(root / 'results/x'), '--seed', '130363', '--sampling-seed', '2'])
        config, seed, sampling, limit_s = settings(args)
        self.assertEqual((seed, sampling, limit_s), (130363, 2, 900.))
        args = parse(base + ['--output', str(root / 'results/x')])
        self.assertEqual(settings(args)[1:3], (104729, 104729))
        with self.assertRaises(ValueError): parse(base + ['--output', str(root / 'x')])

    def test_parse_goal_turns_on_oversight(self):
        from minecraftrl.pipeline.cli import parse
        root = Path(__file__).resolve().parents[1]
        base = ['probe', '--root', str(root), '--config', str(root / 'configs/iron-demo.json'), '--plan-json', 'p.json',
                '--output', str(root / 'results/x')]
        self.assertIsNone(parse(base).goal)
        args = parse(base + ['--goal', 'iron_ore:1', '--kb', str(root / 'results/x/kb.sqlite')])
        self.assertEqual(args.goal, ('iron_ore', 1))
        for bad in (['--goal', 'iron_ore'], ['--goal', 'iron_ore:0'], ['--goal', 'log:3', '--kb', str(root / 'kb.sqlite')]):
            with self.assertRaises(ValueError): parse(base + bad)

    def test_parse_utterance_needs_paid_calls_and_excludes_goal(self):
        from minecraftrl.pipeline.cli import parse, request_goal
        root = Path(__file__).resolve().parents[1]
        base = ['probe', '--root', str(root), '--config', str(root / 'configs/iron-demo.json'), '--plan-json', 'p.json',
                '--output', str(root / 'results/x')]
        self.assertEqual(parse(base + ['--utterance', '철 캐줘', '--allow-paid']).utterance, '철 캐줘')
        for bad in (['--utterance', '철 캐줘'], ['--utterance', '철 캐줘', '--allow-paid', '--goal', 'iron_ore:1'], ['--utterance', ' ', '--allow-paid']):
            with self.assertRaises(ValueError): parse(base + bad)
        config = json.loads((root / 'configs/iron-demo.json').read_text(encoding='utf-8'))
        self.assertEqual((config['allowed_goals'], config['success_criterion']), ({'iron_ore': [1, 64]}, {'item': 'iron_ore', 'count': 1}))
        from test_pipeline_goal import client, reply
        llm, _ = client(reply('iron_ore', None))
        goal = request_goal(llm, '철 캐줘', config)
        self.assertEqual((goal['item'], goal['count'], goal['count_source']), ('iron_ore', 1, 'caller_success_criterion'))

    def test_oversight_preloads_every_skill_a_replan_can_use(self):
        # Real run A (2026-11-04): the replanned chain needed wood_visual and the recipe catalog, which were never loaded.
        from minecraftrl.pipeline.cli import runtime_needs
        config = {'selected': {'log': 'wood_visual', 'cobblestone': 'mine_stone_text', 'iron_ore': 'mine_iron_text'}}
        steps = [Step('mine-iron_ore', 'mine_iron_text', 'policy', 'iron_ore', 1, block='iron_ore')]
        self.assertEqual(runtime_needs(steps, config, oversight=False), (['mine_iron_text'], False))
        self.assertEqual(runtime_needs(steps, config, oversight=True), (['mine_iron_text', 'mine_stone_text', 'wood_visual'], True))

    def test_initial_plan_comes_from_the_knowledge_store_when_no_plan_file(self):
        # Spec 5.12: the inherited-KB condition starts from the full plan; an empty store gives the naive plan.
        import tempfile
        from minecraftrl.pipeline.cli import initial_steps, parse
        from minecraftrl.pipeline.kb import KnowledgeBase
        from test_pipeline_planner import iron_kb
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/iron-demo.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as d:
            with KnowledgeBase(Path(d) / 'empty.sqlite') as empty:
                naive, steps = initial_steps(empty, config, ('iron_ore', 1))
                self.assertTrue(naive.naive); self.assertEqual([(s.skill, s.params) for s in steps], [('mine_iron_text', {'precondition': 'unverified'})])
            with KnowledgeBase(Path(d) / 'iron.sqlite') as kb:
                full, steps = initial_steps(iron_kb(kb), config, ('iron_ore', 1))
                self.assertFalse(full.naive)
                self.assertEqual([s.skill for s in steps], ['wood_visual', 'craft', 'mine_stone_text', 'craft', 'mine_iron_text'])
        base = ['probe', '--root', str(root), '--config', str(root / 'configs/iron-demo.json'), '--output', str(root / 'results/x')]
        self.assertIsNone(parse(base + ['--goal', 'iron_ore:1']).plan_json)
        with self.assertRaises(ValueError): parse(base)  # no plan file and no goal: nothing to plan from

    def test_load_plan_builds_steps(self):
        from minecraftrl.pipeline.cli import load_plan
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plan.json'
            path.write_text(json.dumps([dict(id='s1', skill='wood', kind='policy', target='log', count=3, block='log'),
                                        dict(id='c1', skill='craft', kind='rule', target='wooden_pickaxe', count=1,
                                             params={'goals': ['wooden_pickaxe'], 'equip': 'wooden_pickaxe'})]))
            steps = load_plan(path)
            self.assertEqual([s.kind for s in steps], ['policy', 'rule']); self.assertEqual(steps[1].params['equip'], 'wooden_pickaxe')
            path.write_text(json.dumps([dict(id='x', skill='a', kind='api', target='log', count=1)]))
            with self.assertRaises(ValueError): load_plan(path)


class ConditionTests(unittest.TestCase):
    def test_set_condition_resets_agent_and_returns_metadata(self):
        policy = FakePolicy(); embedding = SKILLS['wood']['embedding']
        meta = set_condition(policy, embedding, 6.)
        self.assertIs(policy.goal_embed, embedding); self.assertEqual(policy.agent.scales, [6.])
        self.assertEqual(meta['cond_scale'], 6.); self.assertEqual(len(meta['embedding_sha256']), 64)

    def test_fixture_sha52_matches_java_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'f.json'; path.write_bytes(b'{}')
            # sha256(b'{}') = 44136fa355b3678a...; first 13 hex digits as an integer.
            self.assertEqual(fixture_sha52(path), int('44136fa355b36', 16))


if __name__ == '__main__':
    unittest.main()
