import importlib.util
import tempfile
import unittest
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from minecraftrl.rl_core import ShapedWoodRewardV4


def gui(tick=0):
    return dict(server_available=1, error=0, server_tick=tick, held_key_count=0, screen=0,
                server_x=6.7, server_y=201., server_z=3.5, yaw=-90., pitch=0.,
                server_inventory={'oak_log':0}, wood_v3_kind=1, wood_v3_distance=5.,
                wood_v3_reachable=0, wood_v3_target_x=11, wood_v3_target_y=202, wood_v3_target_z=3)


class RecordedActor:
    """Only the costly learned actor is substituted; detector and reward are real."""
    def __init__(self):
        self.rows, self.inputs = [], []
        self.resets = 0
        self.agent = SimpleNamespace(hidden_state={'commits':0})
        self.goal_embed = np.zeros((1,512), dtype=np.float32)
        self.condition_metadata = {}

    def reset(self, goal):
        self.resets += 1
        self.condition_metadata = dict(prompt=goal)

    def set_reward_context(self, context):
        self.context = list(context)

    def act(self, screen):
        self.inputs.append(self.goal_embed.copy())
        # A wrapper must not replace this with a neutral or a scripted turn.
        return {'attack':1, 'camera':[0., 0.]}

    def commit(self, action, reward, terminal):
        self.rows.append(dict(action=deepcopy(action)))
        self.agent.hidden_state = {'commits':len(self.rows)}


class GoalRecoveryTests(unittest.TestCase):
    def make_policy(self, *, search=False):
        self.assertIsNotNone(importlib.util.find_spec('minecraftrl.goal_recovery'),
                             'bounded neural goal recovery is not implemented')
        from minecraftrl.goal_recovery import GoalRecoveryController
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        path = folder/'explore.npy'
        np.save(path, np.ones((1,512), dtype=np.float32))
        policy = GoalRecoveryController(RecordedActor(), path, folder)
        initial = gui()
        if search:
            initial.update(wood_v3_kind=0, wood_v3_distance=0.)
        policy.environment = SimpleNamespace(gui=initial, reward=ShapedWoodRewardV4(initial))
        policy.set_reward_context(policy.environment.reward.context())
        policy.reset('collect wood')
        return policy

    def commit_step(self, p, action, changes=None, terminal=False):
        after = deepcopy(p.environment.gui)
        after['server_tick'] += 1
        after.update(changes or {})
        components = p.environment.reward.advance(after, terminal=terminal)
        p.environment.gui = after
        p.commit(action, components['total'], terminal or p.environment.reward.completed)
        p.rows[-1]['reward_components'] = components
        p.set_reward_context(p.environment.reward.context())

    def step(self, p, changes=None):
        action = p.act(np.zeros((2,2,3), dtype=np.uint8))
        self.commit_step(p, action, changes)
        return action

    def trigger(self, p, count=200):
        for _ in range(count):
            self.step(p)

    def test_stall_changes_only_embedding_without_reset_or_forced_action(self):
        p = self.make_policy()
        reward = p.environment.reward
        self.trigger(p)
        self.assertEqual(p.switches, [])  # No new action has consumed the request.
        action = self.step(p)
        self.assertEqual(action, {'attack':1, 'camera':[0.,0.]})
        self.assertEqual(p.rows[-1]['mode'], 'intervention')
        np.testing.assert_array_equal(p.student.inputs[-1], np.ones((1,512)))
        self.assertEqual(p.student.resets, 1)
        self.assertEqual(p.student.agent.hidden_state, {'commits':201})
        self.assertIs(p.environment.reward, reward)
        self.assertTrue(p.switches[0]['committed'])
        self.assertEqual(p.switches[0]['after_step'], 200)

    def test_search_stall_also_uses_graph_detector(self):
        p = self.make_policy(search=True)
        self.trigger(p, 399)
        self.assertIsNone(p.trigger)
        self.step(p)
        self.assertEqual(p.trigger['step'], 400)
        self.step(p)
        self.assertEqual(p.rows[-1]['mode'], 'intervention')

    def test_budget_returns_to_wood_once_and_never_rearms(self):
        p = self.make_policy()
        self.trigger(p)
        for _ in range(200):
            self.step(p)
        self.assertEqual(sum(r['mode']=='intervention' for r in p.rows), 200)
        self.step(p)
        self.assertEqual(p.rows[-1]['mode'], 'wood')
        np.testing.assert_array_equal(p.student.inputs[-1], np.zeros((1,512)))
        for _ in range(410):
            self.step(p)
        self.assertEqual(len(p.switches), 2)
        self.assertEqual(p.switches[-1]['reason'], 'tick_budget')

    def test_new_view_is_not_completion_but_reachable_wood_returns(self):
        p = self.make_policy()
        self.trigger(p)
        self.step(p, dict(yaw=10., server_z=6., wood_v3_distance=6.))
        self.step(p)
        self.assertEqual(p.rows[-1]['mode'], 'intervention')
        self.step(p, dict(wood_v3_reachable=1, wood_v3_distance=2.))
        self.step(p)
        self.assertEqual(p.rows[-1]['mode'], 'wood')
        self.assertEqual(p.switches[-1]['reason'], 'wood_interaction_ready')
        self.assertFalse(p.environment.reward.completed)

    def test_stop_after_inference_commits_old_action_without_starting_intervention(self):
        p = self.make_policy()
        self.trigger(p, 199)
        action = p.act(np.zeros((2,2,3), dtype=np.uint8))
        p.is_stopped = lambda: True
        self.commit_step(p, action)
        self.assertEqual(len(p.rows), 200)
        self.assertEqual(p.switches, [])
        with self.assertRaises(InterruptedError):
            p.act(None)
        self.assertEqual(len(p.student.inputs), 200)

    def test_acceptance_wins_over_stall_and_forbids_later_inference(self):
        p = self.make_policy()
        self.trigger(p, 199)
        self.step(p, dict(server_inventory={'oak_log':1}))
        self.assertTrue(p.environment.reward.completed)
        self.assertEqual(p.switches, [])
        with self.assertRaises(InterruptedError):
            p.act(None)
        self.assertEqual(len(p.student.inputs), 200)

    def test_dispatch_mismatch_does_not_commit_either_trajectory(self):
        p = self.make_policy()
        p.act(np.zeros((2,2,3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            p.commit({'attack':0},0.,False)
        self.assertEqual(p.rows, [])
        self.assertEqual(p.student.rows, [])

    def test_unsupported_gui_rejects_switch_before_new_policy_inference(self):
        p = self.make_policy()
        self.trigger(p)
        p.environment.gui['screen'] = 1
        with self.assertRaisesRegex(ValueError, 'GUI'):
            p.act(None)
        self.assertEqual(len(p.student.inputs), 200)
        self.assertEqual(p.switches, [])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'trajectory serialization requires torch')
    def test_collector_saves_ineligible_diagnostics_and_stops_at_inventory_acceptance(self):
        import minecraftrl.goal_recovery as module
        self.assertTrue(hasattr(module, 'run_goal_recovery'), 'diagnostic collector is not implemented')
        import torch
        from minecraftrl.runtime import StopGate
        p = self.make_policy()
        # run_episode owns the sole reset in this separate collector scenario.
        p.student.resets = 0
        p.student.torch = torch
        p.student.initial_heads = dict(actor={'a':torch.zeros(1)}, value={'v':torch.zeros(1)})
        p.student.initial_state = {'commits':0}
        p.student.dispose = lambda: None

        class Environment:
            raw = SimpleNamespace(action_space=SimpleNamespace(noop=lambda: {'attack':0,'camera':[0.,0.]}))
            release_evidence = dict(verified=True, held_key_count=0)
            def __init__(self): self.state = gui(); self.releases = 0; self.closed = False
            def seed(self, seed): pass
            def reset(self): pass
            def step(self, action):
                self.state['server_tick'] += 1
                if self.state['server_tick'] == 63:
                    self.state['server_inventory'] = {'oak_log':1}
                return {'pov':np.zeros((16,16,3),dtype=np.uint8),
                        'inventory':deepcopy(self.state['server_inventory'])},0.,False,{'workshop':deepcopy(self.state)}
            def release(self): self.releases += 1; return 'neutral_command_sent'
            def close(self): self.closed = True

        env = Environment()
        report = module.run_goal_recovery(p,env,p.output,world={'seed':1},split='eval',
                    seed=1,limit_s=5,gate=StopGate(),stop_file=p.output/'STOP')
        self.assertTrue(report['server_target_reached'], report)
        self.assertFalse(report['rl_training_eligible'])
        self.assertEqual(report['steps'],3)
        self.assertEqual(len(p.student.inputs),3)
        self.assertEqual(env.releases,1)
        self.assertTrue(env.closed)
        saved = torch.load(p.output/'trajectory.pt',map_location='cpu',weights_only=False)
        self.assertFalse(saved['eligible'])
        self.assertEqual(saved['scope'],'goal_conditioning_diagnostic')
        self.assertEqual(len(saved['rows']),3)
        self.assertFalse(any(row['loss_eligible'] for row in json.loads((p.output/'goal-recovery-rows.json').read_text())))


if __name__ == '__main__':
    unittest.main()
