"""Where the time goes: one horizontal bar per demo run, split into phases (recorded clock).

Usage: python scripts/plot_demo_timelines.py OUT.png RUN_DIR [RUN_DIR ...] [--title TEXT]
Phases (fixed categorical order): first iron search (naive plan), hold + slow path (SQLite/LLM wait), harvesting
(logs/cobblestone), rule skills (craft/equip/clear/turn-away), pit escape (climb), iron mining (final step).
The bar end carries the server-judged result. Colors are the dataviz reference slots 1-6 (validated, light surface);
three sit below 3:1 contrast, so the legend and the numbers table in the results document carry identity too.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PHASES = [('first', '철광석 첫 탐색 (지식 없음)', '#2a78d6'), ('hold', '정지 · 지식 조회/LLM', '#eb6834'),
          ('harvest', '원목 · 조약돌 (정책)', '#1baf7a'), ('rule', '규칙 스킬 (제작·장착)', '#eda100'),
          ('escape', '구덩이 탈출 (정책)', '#e87ba4'), ('iron', '철광석 채굴 (정책)', '#008300')]
COLOR = {k: c for k, _, c in PHASES}
TEXT, MUTED, GRID, SURFACE = '#1d1d1f', '#6b6b70', '#e3e1dc', '#fcfcfb'


def read_jsonl(path): return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def phase_of(e, seen_iron):
    if e.get('id') == 'mine-iron_ore': return 'iron' if seen_iron else 'first'
    if e.get('step_kind') == 'rule' or e.get('skill') in ('craft', 'equip', 'clear_above'): return 'rule'
    if str(e.get('id', '')).startswith('climb'): return 'escape'
    return 'harvest'


def segments(run):
    ev = read_jsonl(run / 'timeline.jsonl'); starts = [e for e in ev if e['kind'] == 'step_started']
    if not starts: return [], 0., 0.
    t0 = starts[0]['t']; out = []; current = None
    # Only a naive start (empty store) has a "first search" step; an inherited store starts with the full plan.
    first_plan = next((e.get('plan') for e in ev if e['kind'] == 'session_started'), None) or []
    initial = next((e for e in ev if e['kind'] == 'initial_plan'), None)
    naive = initial['naive'] if initial else bool(first_plan and (first_plan[0].get('params') or {}).get('precondition') == 'unverified')
    seen_iron = not naive
    for e in ev:
        if e['kind'] == 'step_started':
            current = [e['t'], phase_of(e, seen_iron)]
            if e.get('id') == 'mine-iron_ore' and not seen_iron: seen_iron = True
        elif e['kind'] == 'hold' and current:
            out.append((current[0] - t0, e['t'] - current[0], current[1])); current = [e['t'], 'hold']
        elif e['kind'] == 'step_finished' and current:
            out.append((current[0] - t0, e['t'] - current[0], current[1])); current = None
    end = max(e['t'] for e in ev) - t0
    if current: out.append((current[0] - t0, end - (current[0] - t0), current[1]))
    return out, end, t0


def main():
    p = argparse.ArgumentParser(); p.add_argument('out', type=Path); p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--title', default='실행별 시간 사용 (서버 판정 결과)'); a = p.parse_args()
    font = font_manager.FontProperties(fname='C:/Windows/Fonts/malgunbd.ttf')
    plt.rcParams['font.family'] = font.get_name(); font_manager.fontManager.addfont('C:/Windows/Fonts/malgunbd.ttf')
    rows = []
    for run in a.runs:
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        segs, end, _ = segments(run)
        iron = (summary.get('final_inventory') or {}).get('iron_ore', 0)
        result = '성공' if summary.get('execution_valid') and iron >= 1 else {
            'time_limit': '시간 초과', 'anomaly_hold': '정지', 'goal_rejected': '시작 전 실패'}.get(summary.get('status'), summary.get('status'))
        rows.append((run.name, segs, end, result))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for i, (name, segs, end, result) in enumerate(rows):
        y = len(rows) - 1 - i
        for start, width, phase in segs:
            if width <= 0: continue
            ax.barh(y, width, left=start, height=0.62, color=COLOR[phase], edgecolor=SURFACE, linewidth=1.0)
        ax.text(end + 8, y, result, va='center', fontsize=8.5, color=TEXT if result == '성공' else MUTED)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5, color=TEXT)
    ax.axvline(900, color=MUTED, linewidth=1, linestyle=(0, (3, 3))); ax.text(902, len(rows) - 0.4, '제한 900초', fontsize=8, color=MUTED)
    ax.set_xlim(0, 1080); ax.set_xlabel('세션 시작부터 초 (기록 시계)', fontsize=9, color=MUTED)
    ax.grid(axis='x', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID); ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_title(a.title, fontsize=11, color=TEXT, loc='left')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PHASES]
    ax.legend(handles, [label for _, label, _ in PHASES], loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.6 / len(rows)),
              ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, facecolor=SURFACE)
    print(a.out)


if __name__ == '__main__':
    main()
