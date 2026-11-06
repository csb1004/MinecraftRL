"""Offline split of recorded policy steps into finer skill stages (plan 2026-11-07 task A; no game, GPU or API).

Each policy row of a step with a target block gets one stage, by priority:
gui > mining > aimed > collect > near > closing > far.
- mining: the server reports mining progress on the target block family.
- aimed: the crosshair (server hit block) is on the target family.
- collect: a target block broke within COLLECT_TICKS and the target item has not increased since.
- near / closing / far: by the eye's distance to the nearest remaining target block of the fixture.

Target positions come from the run's own fixture config and start pose. This is privileged server geometry for
offline analysis only, not a policy input. Recorded hit positions check the geometry; a target block hit where
the fixture has none was placed by the player and joins the step's geometry. Seconds are server ticks / 20,
so slow server phases and holds (frozen world) do not count.
"""
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

from .monitor_replay import load_run, policy_rows, run_label, sample_from_row
from .steps import block_matches, item_count

STAGES = ('far', 'closing', 'near', 'aimed', 'mining', 'collect', 'gui')
REACH = 4.5                          # blocks from the eye to a block centre
EYE = 1.62
CLOSING_TICKS, CLOSING_BLOCKS = 20, .5
COLLECT_TICKS = 100
MAX_ROW_TICKS = 20                   # a longer gap between rows counts as this many ticks
PIT_DEPTH = .5                       # feet more than this below the fixture origin = in a pit
FACINGS = ((0, 1), (-1, 0), (0, -1), (1, 0))   # Minecraft horizontal index: south, west, north, east as (x, z)


def facing_index(yaw):
    """Direction.fromAngle: floor(yaw / 90 + 0.5) & 3."""
    return int(math.floor(yaw / 90. + .5)) & 3


def family(block):
    return 'log' if block.endswith('_log') else block


def fixture_blocks(fixture, origin, yaw):
    """Block per position as DemoFixture places it: fills in order, then single blocks; dx right of the start
    facing, dz forward, dy up. Without rotate_with_facing the facing is south."""
    i = facing_index(yaw) if fixture.get('rotate_with_facing') else 0
    (fx, fz), (rx, rz) = FACINGS[i], FACINGS[(i + 1) % 4]
    ox, oy, oz = origin

    def at(dx, dy, dz): return ox + fx * dz + rx * dx, oy + dy, oz + fz * dz + rz * dx
    out = {}
    for region in fixture.get('fill', []):
        low = [min(a, b) for a, b in zip(region['from'], region['to'])]
        high = [max(a, b) for a, b in zip(region['from'], region['to'])]
        for dx in range(low[0], high[0] + 1):
            for dy in range(low[1], high[1] + 1):
                for dz in range(low[2], high[2] + 1): out[at(dx, dy, dz)] = region['block']
    for b in fixture.get('blocks', []): out[at(b['dx'], b['dy'], b['dz'])] = b['block']
    return out


def fixture_origin(rows):
    """(start block, yaw) at the first row that reports the fixture applied. y is the fixture's own origin; records
    from jars before that field use the feet block, which is where DemoFixture puts the origin."""
    for r in rows:
        d = r.get('demo') or {}
        if d.get('demo_fixture_state') == 2:
            y = d.get('demo_fixture_origin_y')
            y = math.floor(r['server_y']) if y is None else int(y)
            return (math.floor(r['server_x']), y, math.floor(r['server_z'])), float(r['yaw'])
    return None


class Geometry:
    """Remaining fixture blocks by family; broken blocks are removed."""

    def __init__(self, blocks, origin_y=None):
        self.blocks = dict(blocks); self.origin_y = origin_y
        self.by_family = defaultdict(set)
        for pos, block in self.blocks.items(): self.by_family[family(block)].add(pos)

    def copy(self):
        return Geometry(self.blocks, self.origin_y)

    def add(self, pos, block):
        self.remove(pos); self.blocks[tuple(pos)] = block; self.by_family[family(block)].add(tuple(pos))

    def remove(self, pos):
        block = self.blocks.pop(tuple(pos), None)
        if block is not None: self.by_family[family(block)].discard(tuple(pos))

    def knows(self, pos, block):
        known = self.blocks.get(tuple(pos))
        return known is not None and family(known) == family(block)

    def nearest(self, wanted, eye):
        found = self.nearest_block(wanted, eye)
        return None if found is None else found[0]

    def nearest_block(self, wanted, eye):
        """(distance from the eye to the block centre, block position) of the nearest remaining block, or None."""
        positions = self.by_family.get(family(wanted)) or ()
        if not positions: return None
        ex, ey, ez = eye
        d2, pos = min(((x + .5 - ex) ** 2 + (y + .5 - ey) ** 2 + (z + .5 - ez) ** 2, (x, y, z)) for x, y, z in positions)
        return math.sqrt(d2), pos


ANGLE_BINS = ((15., '<15'), (45., '15-45'), (90., '45-90'), (math.inf, '>=90'))


def view_angle(yaw, pitch, eye, pos):
    """Degrees between the view direction (Minecraft yaw 0 = +z, 90 = -x; pitch down positive) and the block centre."""
    y, p = math.radians(yaw), math.radians(pitch)
    view = (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))
    to = (pos[0] + .5 - eye[0], pos[1] + .5 - eye[1], pos[2] + .5 - eye[2])
    norm = math.sqrt(sum(c * c for c in to)) or 1.
    return math.degrees(math.acos(max(-1., min(1., sum(a * b / norm for a, b in zip(view, to))))))


def destroy_events(rows):
    """(time, position, block) whenever the server destroy counter rises between valid rows."""
    events, prev = [], None
    for r in rows:
        s = sample_from_row(r)
        if not s.valid: continue
        if prev is not None and s.destroys > prev and s.last_destroyed_pos is not None:
            events.append((r['t'], s.last_destroyed_pos, s.last_destroyed))
        prev = s.destroys
    return events


def _stage(s, wanted, distance, closing, collecting):
    if s.gui_open: return 'gui'
    if block_matches(s.mining_block, wanted): return 'mining'
    if block_matches(s.hit_block, wanted): return 'aimed'
    if collecting: return 'collect'
    if distance is not None and distance <= REACH: return 'near'
    if closing: return 'closing'
    return 'far'


def step_stages(run, step, geometry, events=None):
    """Seconds per stage for the policy rows of one step (geometry: the fixture at the start of the run)."""
    wanted = step.block or step.target
    events = destroy_events(run.rows) if events is None else events
    rows = [(r, s) for r, s in ((r, sample_from_row(r)) for r in policy_rows(run, step)) if s.valid]
    g, pending = geometry.copy(), deque(events)
    seconds = dict.fromkeys(STAGES, 0.)
    pit, checked, mismatch = 0., 0, 0
    recent, collect = deque(), None
    near_hits, broken = defaultdict(float), {'level_or_above': 0, 'below': 0}
    near_angle = {label: 0. for _, label in ANGLE_BINS}
    for i, (r, s) in enumerate(rows):
        origin_y = (r.get('demo') or {}).get('demo_fixture_origin_y', g.origin_y)
        while pending and pending[0][0] <= r['t']:
            t, pos, block = pending.popleft(); g.remove(pos)
            if t >= step.start_t and block_matches(block, wanted):
                collect = (s.tick, item_count(s.inventory, step.target))
                if origin_y is not None: broken['below' if pos[1] < origin_y else 'level_or_above'] += 1
        if block_matches(s.hit_block, wanted) and s.hit_pos is not None:
            checked += 1; mismatch += not geometry.knows(s.hit_pos, s.hit_block)
            if not g.knows(s.hit_pos, s.hit_block): g.add(s.hit_pos, s.hit_block)   # placed by the player
        eye = (s.position[0], s.position[1] + EYE, s.position[2])
        nearest = g.nearest_block(wanted, eye)
        distance = None if nearest is None else nearest[0]
        while len(recent) > 1 and recent[1][0] <= s.tick - CLOSING_TICKS: recent.popleft()
        ref = recent[0] if recent and recent[0][0] <= s.tick - CLOSING_TICKS else None
        recent.append((s.tick, distance))
        closing = (ref is not None and ref[1] is not None and distance is not None and ref[1] - distance > CLOSING_BLOCKS)
        collecting = (collect is not None and s.tick - collect[0] <= COLLECT_TICKS
                      and item_count(s.inventory, step.target) <= collect[1])
        dt = min(max(rows[i + 1][1].tick - s.tick, 0), MAX_ROW_TICKS) / 20. if i + 1 < len(rows) else 0.
        stage = _stage(s, wanted, distance, closing, collecting)
        seconds[stage] += dt
        if stage == 'near':
            near_hits[s.hit_block] += dt
            angle = view_angle(s.yaw, s.pitch, eye, nearest[1])
            near_angle[next(label for limit, label in ANGLE_BINS if angle < limit)] += dt
        if origin_y is not None and s.position[1] < origin_y - PIT_DEPTH: pit += dt
    first = rows[0][1].inventory if rows else {}
    return dict(run=run_label(run), step=step.id, skill=step.skill, target=wanted, status=step.status,
                tool_at_start=sorted(k for k in first if k.endswith('_pickaxe')), seconds=seconds,
                total_s=sum(seconds.values()), pit_s=pit, rows=len(rows), geometry_checked=checked,
                geometry_mismatch=mismatch, near_hits=dict(near_hits), near_angle_s=near_angle, target_broken=broken)


def _resolve(recorded, root):
    """The recorded fixture path, or the same file under `root` when the record came from another mount."""
    path = Path(recorded)
    if path.exists(): return path
    parts = path.parts
    for i in range(1, len(parts)):
        candidate = Path(root).joinpath(*parts[i:])
        if candidate.exists(): return candidate
    return None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome(status):
    return {'completed': 'completed', 'time_limit': 'time_limit', 'anomaly_hold': 'hold'}.get(status, status)


def _groups(steps):
    groups = defaultdict(list)
    for s in steps:
        groups[(s['skill'], '+'.join(s['tool_at_start']) or 'no_tool', outcome(s['status']))].append(s)
    out = []
    for (skill, tool, result), members in sorted(groups.items()):
        seconds = {k: sum(m['seconds'][k] for m in members) for k in STAGES}
        near = Counter()
        for m in members: near.update(m['near_hits'])
        out.append(dict(skill=skill, tool=tool, outcome=result, n=len(members), seconds=seconds,
                        total_s=sum(seconds.values()), pit_s=sum(m['pit_s'] for m in members),
                        median_total_s=statistics.median(m['total_s'] for m in members),
                        near_hits=dict(near.most_common()),
                        near_angle_s={k: sum(m['near_angle_s'][k] for m in members) for _, k in ANGLE_BINS},
                        target_broken={k: sum(m['target_broken'][k] for m in members) for k in ('level_or_above', 'below')}))
    return out


def build_report(run_dirs, *, root):
    """Stage seconds for every policy step with a target block. A run whose fixture file is missing or differs
    from the recorded sha256, or that never reports the fixture applied, is skipped with a reason."""
    steps, skipped, inputs = [], [], []
    for d in map(Path, run_dirs):
        if not ((d / 'timeline.jsonl').exists() and (d / 'telemetry.jsonl').exists()):
            skipped.append(dict(dir=str(d), reason='no records')); continue
        request = json.loads((d / 'request.json').read_text(encoding='utf-8')) if (d / 'request.json').exists() else {}
        fixture_path = _resolve(request['fixture'], root) if request.get('fixture') else None
        if fixture_path is None: skipped.append(dict(dir=str(d), reason='no fixture file')); continue
        if sha256(fixture_path) != request.get('fixture_sha256'):
            skipped.append(dict(dir=str(d), reason='fixture sha256 differs from the record')); continue
        run = load_run(d)
        start = fixture_origin(run.rows)
        if start is None: skipped.append(dict(dir=str(d), reason='fixture never reported applied')); continue
        geometry = Geometry(fixture_blocks(json.loads(fixture_path.read_text(encoding='utf-8')), *start), start[0][1])
        events = destroy_events(run.rows)
        inputs.append(dict(dir=str(d), seed=run.seed, fixture=str(fixture_path), origin=list(start[0]), yaw=start[1]))
        steps += [step_stages(run, s, geometry, events) for s in run.steps if s.block]
    here = Path(__file__).resolve()
    return dict(constants=dict(reach=REACH, eye=EYE, closing_ticks=CLOSING_TICKS, closing_blocks=CLOSING_BLOCKS,
                               collect_ticks=COLLECT_TICKS, max_row_ticks=MAX_ROW_TICKS, pit_depth=PIT_DEPTH),
                source={'stage_replay.py': sha256(here)}, inputs=inputs, skipped=skipped, steps=steps,
                groups=_groups(steps),
                geometry=dict(checked=sum(s['geometry_checked'] for s in steps),
                              mismatch=sum(s['geometry_mismatch'] for s in steps)))


def report_markdown(report):
    lines = ['# 하위 단계 시간 분해', '',
             f"- 기록 {len(report['inputs'])}개, 건너뜀 {len(report['skipped'])}개. 시간은 서버 틱 / 20(게임 초).",
             f"- 대상 위치 확인: 대상 블록 조준 {report['geometry']['checked']}행 중 fixture에 없는 위치 "
             f"{report['geometry']['mismatch']}행.", '',
             '| 스킬 | 도구 | 결과 | 단계 수 | 합계(s) | 중앙값(s) | ' + ' | '.join(STAGES) + ' | 구덩이 |',
             '|---|---|---|---|---|---|' + '---|' * (len(STAGES) + 1)]
    for g in report['groups']:
        total = g['total_s'] or 1.
        shares = ' | '.join(f"{100 * g['seconds'][k] / total:.0f}%" for k in STAGES)
        lines.append(f"| {g['skill']} | {g['tool']} | {g['outcome']} | {g['n']} | {g['total_s']:.0f} | "
                     f"{g['median_total_s']:.0f} | {shares} | {100 * g['pit_s'] / total:.0f}% |")
    lines += ['', '## near 동안 조준선 블록, 부순 대상의 높이', '',
              '| 스킬 | 도구 | 결과 | near 조준선 상위 3 (s) | near 시선-대상 각도 <15 / 15-45 / 45-90 / >=90 | '
              '부순 대상: 서 있는 높이 이상 / 아래 |', '|---|---|---|---|---|---|']
    for g in report['groups']:
        top = ', '.join(f'{b} {v:.0f}' for b, v in list(g['near_hits'].items())[:3]) or '-'
        near = sum(g['near_angle_s'].values()) or 1.
        angles = ' / '.join(f"{100 * g['near_angle_s'][k] / near:.0f}%" for _, k in ANGLE_BINS)
        lines.append(f"| {g['skill']} | {g['tool']} | {g['outcome']} | {top} | {angles} | "
                     f"{g['target_broken']['level_or_above']} / {g['target_broken']['below']} |")
    return '\n'.join(lines) + '\n'
