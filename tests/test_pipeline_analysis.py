import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.analysis import analyze_batch, analyze_run


def write_run(root, name, *, iron, status='completed', valid=True, destroys=(), holds=(), resumes=(), api_tokens=(), masks=(3,)):
    run = root / name; run.mkdir(parents=True)
    rows = []
    for step in range(1, 6):
        inv = {'iron_ore': iron} if step == 5 else {}
        demo = dict(demo_destroys=0)
        for i, (at, block, harvestable) in enumerate(destroys):
            if step >= at: demo.update(demo_destroys=i + 1, demo_last_destroyed_block=block, demo_last_destroy_harvestable=harvestable)
        rows.append(dict(t=float(step), step=step, controller='policy', inventory=inv, demo=demo))
    (run / 'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    events = [dict(t=0., step=0, kind='session_started', plan=[])]
    events += [dict(t=1., step=1, kind='hold', reason=r, hold_latency_ms=ms, mining_ticks_at_hold=tk) for r, ms, tk in holds]
    events += [dict(t=2., step=1, kind='resume', resolution='replan', detect_to_resume_s=s) for s in resumes]
    (run / 'timeline.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    stages = [dict(id='mine-iron_ore', status=status, attack_masks=m, steps=4) for m in masks]
    summary = dict(status=status, execution_valid=valid, elapsed_control_s=300., steps=5, final_inventory={'iron_ore': iron},
                   release='neutral_command_sent', stage_results=stages,
                   oversight=dict(api_calls=len(api_tokens), usage=[dict(total_tokens=t) for t in api_tokens]))
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return run


class AnalysisTests(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_success_is_recomputed_from_server_telemetry(self):
        run = write_run(self.root, 'r1', iron=1, holds=[('guard_unknown', .45, 0), ('rule_skill_failed', .5, None)],
                        resumes=[26.9, 14.0], api_tokens=[10000, 9000])
        a = analyze_run(run)
        self.assertEqual((a['success'], a['iron_ore'], a['api_calls'], a['tokens']), (True, 1, 2, 19000))
        self.assertEqual((a['hold_latency_ms'], a['guard_mining_ticks'], a['detect_to_resume_s']), ([.45, .5], [0], [26.9, 14.0]))
        self.assertEqual((a['attack_masks'], a['guarded_block_losses'], a['failure_category']), (3, 0, None))

    def test_bare_hand_iron_loss_is_counted_and_failures_are_categorised(self):
        lost = analyze_run(write_run(self.root, 'r2', iron=0, status='time_limit', destroys=[(3, 'iron_ore', 0), (4, 'stone', 0)]))
        self.assertEqual((lost['success'], lost['guarded_block_losses'], lost['failure_category']), (False, 1, 'motor'))
        held = analyze_run(write_run(self.root, 'r3', iron=0, status='anomaly_hold'))
        self.assertEqual(held['failure_category'], 'knowledge')
        broken = analyze_run(write_run(self.root, 'r4', iron=0, status='error', valid=False))
        self.assertEqual(broken['failure_category'], 'technical')

    def test_held_failures_are_split_by_what_stopped_them(self):
        # Evaluation round 1: E01 stopped at the retry limit of a policy fixation (motor); E03 stopped because the API call
        # failed (technical); only an unusable answer or plan is a knowledge failure.
        def held(name, slow):
            run = write_run(self.root, name, iron=0, status='anomaly_hold')
            summary = json.loads((run / 'summary.json').read_text()); summary['stage_results'][-1]['slow_path'] = slow
            (run / 'summary.json').write_text(json.dumps(summary)); return analyze_run(run)['failure_category']
        self.assertEqual(held('h1', dict(route='retry_limit', llm='not_needed', resolution='stop')), 'motor')
        self.assertEqual(held('h2', dict(route='acquire_then_replan', llm='failed', resolution='stop')), 'technical')
        self.assertEqual(held('h3', dict(route='acquire_then_replan', llm='called', resolution='stop')), 'knowledge')
        # Round 2 E07: the policy placed its pillar blocks off the column; too few blocks left is a motor outcome.
        self.assertEqual(held('h4', dict(route='replan_from_kb', llm='not_needed', resolution='stop',
                                         reason="replan still naive or has problems: ['not enough spare placeable blocks to pillar 4 levels: 3 of 3']")), 'motor')

    def test_success_by_status_for_scenes_without_an_item_goal(self):
        run = write_run(self.root, 'c1', iron=0, status='completed')
        self.assertEqual((analyze_run(run)['success'], analyze_run(run, by_status=True)['success']), (False, True))

    def test_batch_keeps_conditions_apart(self):
        write_run(self.root / 'empty', 'r1', iron=1); write_run(self.root / 'empty', 'r2', iron=0, status='time_limit')
        write_run(self.root / 'warm', 'w1', iron=1)
        a = analyze_batch({'empty': sorted((self.root / 'empty').iterdir()), 'warm': sorted((self.root / 'warm').iterdir())})
        self.assertEqual((a['empty']['success'], a['empty']['runs'], a['warm']['success'], a['warm']['runs']), (1, 2, 1, 1))
        self.assertNotIn('total', a)  # success rates of the two conditions are never merged (spec 5.12)


if __name__ == '__main__':
    unittest.main()
