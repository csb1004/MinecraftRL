import json
import tempfile
import unittest
from pathlib import Path

from minecraftrl.pipeline.viewer import badge_at, build_replay, events_at, kb_at, rm_at


def claim(cid, status, **kw):
    row = dict(id=cid, subject='iron_ore', relation='harvest_requires', object={'tool_class': 'pickaxe', 'min_tier': 'stone'},
               conditions={'game_version': '1.16.5'}, status=status, revision=1, support=2, refute=0,
               sources=['https://minecraft.wiki/w/Iron_Ore'], observed={'support': 0, 'refute': 0})
    row.update(kw); return row


TIMELINE = [
    dict(t=0., step=0, kind='session_started', plan=[dict(id='mine-iron_ore', skill='mine_iron_text', kind='policy', target='iron_ore', count=1)]),
    dict(t=0.1, step=0, kind='kb_snapshot', reason='session_start', claims=[]),
    dict(t=1., step=2, kind='step_started', id='mine-iron_ore', skill='mine_iron_text', step_kind='policy', target='iron_ore', count=1),
    dict(t=3., step=4, kind='anomaly', step_id='mine-iron_ore',
         event=dict(kind='guard_unknown', hold=True, phase='pre_dispatch', block='iron_ore', mining_ticks=0)),
    dict(t=3., step=4, kind='hold', step_id='mine-iron_ore', reason='guard_unknown', mining_ticks_at_hold=0, hold_latency_ms=0.45),
    dict(t=3., step=4, kind='kb_query', subject='iron_ore', relation='harvest_requires'),
    dict(t=3.1, step=4, kind='llm_request', purpose='acquire_knowledge', model='gpt-5.4'),
    dict(t=14., step=4, kind='kb_change', claim_ids=[1], rejected=[]),
    dict(t=14., step=4, kind='plan', plan_id='plan-x', steps=[
        dict(id='logs', skill='wood_visual', kind='policy', target='log', count=4),
        dict(id='mine-iron_ore', skill='mine_iron_text', kind='policy', target='iron_ore', count=1)], mode='replan'),
    dict(t=14.1, step=4, kind='kb_snapshot', reason='slow_path', claims=[claim(1, 'proposed')]),
    dict(t=14.2, step=4, kind='resume', resolution='replan', detect_to_resume_s=11.2),
    dict(t=15., step=6, kind='step_started', id='logs', skill='wood_visual', step_kind='policy', target='log', count=4),
    dict(t=16., step=7, kind='monitor_event', step_id='logs', event=dict(kind='guard_violation', hold=False, mask_attack=True)),
    dict(t=20., step=10, kind='step_finished', id='logs', status='completed'),
    dict(t=20.1, step=10, kind='kb_snapshot', reason='step_verified', claims=[claim(1, 'supported', observed={'support': 1, 'refute': 0})]),
]


STEP_T = {1: .5, 2: 1., 3: 2., 4: 3., 5: 14.3, 6: 15., 7: 16., 8: 17., 9: 18., 10: 20.}


def telemetry():
    return [dict(t=STEP_T[step], step=step, controller='policy', inventory={'oak_log': max(0, step - 6)}, server_selected_item='air',
                 action={'attack': 1}, demo={'demo_hit_block': 'iron_ore'}) for step in range(1, 11)]


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.run = Path(self.temp.name) / 'run'; (self.run / 'frames').mkdir(parents=True)
        (self.run / 'timeline.jsonl').write_text('\n'.join(json.dumps(e) for e in TIMELINE) + '\n', encoding='utf-8')
        (self.run / 'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in telemetry()) + '\n', encoding='utf-8')
        for step in range(2, 11, 2): (self.run / 'frames' / ('%06d.jpg' % step)).write_bytes(b'\xff\xd8\xff\xd9')
        (self.run / 'summary.json').write_text(json.dumps(dict(status='step_failed', final_inventory={'oak_log': 4})), encoding='utf-8')

    def tearDown(self): self.temp.cleanup()

    def test_kb_at_replays_status_changes(self):
        # Time axis is the recorded clock: a hold freezes the game (no steps) while the slow path runs.
        self.assertEqual(kb_at(TIMELINE, 3.)['claims'], [])
        rows = kb_at(TIMELINE, 14.5); self.assertEqual([(c['id'], c['status']) for c in rows['claims']], [(1, 'proposed')])
        self.assertEqual(rows['changed'], [1])
        later = kb_at(TIMELINE, 20.5); self.assertEqual((later['claims'][0]['status'], later['changed']), ('supported', [1]))
        self.assertEqual(kb_at(TIMELINE, 19.)['changed'], [])  # highlight only for a few seconds after the change

    def test_hold_badge_uses_recorded_latency(self):
        self.assertEqual(badge_at(TIMELINE, 3.)['text'], '⏸ 파괴 전 정지 (0.45 ms · 채굴 경과 0틱)')
        self.assertEqual(badge_at(TIMELINE, 10.)['text'], '⏸ 파괴 전 정지 (0.45 ms · 채굴 경과 0틱)')  # until the resume
        self.assertIsNone(badge_at(TIMELINE, 14.5))

    def test_guard_badges_show_mining_ticks_and_masks(self):
        self.assertEqual(badge_at(TIMELINE, 16.5)['text'], '⛔ 타격 차단'); self.assertIsNone(badge_at(TIMELINE, 18.))
        other = [dict(t=1., step=3, kind='hold', reason='blocked_overhead', hold_latency_ms=1.5, mining_ticks_at_hold=None)]
        self.assertEqual(badge_at(other, 1.)['text'], '⏸ 즉시 정지 (1.50 ms)')

    def test_rm_progress_follows_replans(self):
        self.assertEqual(rm_at(TIMELINE, 2.5), [dict(id='mine-iron_ore', target='iron_ore', count=1, status='active')])
        self.assertEqual(rm_at(TIMELINE, 10.)[0]['status'], 'held')  # from the hold, while the slow path runs
        self.assertEqual([(s['id'], s['status']) for s in rm_at(TIMELINE, 16.)], [('logs', 'active'), ('mine-iron_ore', 'pending')])
        self.assertEqual([(s['id'], s['status']) for s in rm_at(TIMELINE, 20.5)], [('logs', 'done'), ('mine-iron_ore', 'pending')])

    def test_recent_events_are_described_in_korean(self):
        lines = [e['text'] for e in events_at(TIMELINE, 14.5)]
        self.assertIn('지식 조회: iron_ore harvest_requires', lines); self.assertIn('LLM 웹 검색 요청 (gpt-5.4)', lines)
        self.assertIn('재개 (감지→재개 11.2초)', lines)

    def test_html_contains_required_panels_offline(self):
        out = build_replay(self.run, Path(self.temp.name) / 'replay')
        html = out.read_text(encoding='utf-8')
        self.assertEqual(out.name, 'replay.html')
        for text in ('준비된 시연 월드', '주장', '조건', '상태', '근거', 'RM 진행', '방금 일어난 일'): self.assertIn(text, html)
        self.assertNotIn('src="http', html); self.assertNotIn("src='http", html)
        self.assertTrue((out.parent / 'frames' / '000004.jpg').exists())
        data = json.loads((out.parent / 'replay-data.json').read_text(encoding='utf-8'))
        self.assertEqual([f['step'] for f in data['frames']], [2, 4, 6, 8, 10])
        self.assertEqual(data['frames'][1]['badge']['text'], '⏸ 파괴 전 정지 (0.45 ms · 채굴 경과 0틱)')


if __name__ == '__main__':
    unittest.main()
