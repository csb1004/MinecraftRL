"""Sensor-tier repair (user request 2026-11-23, design docs/specs/2026-11-23-sensor-repair-design.md): notice a wrong
block on the sensors, dig it with a pickaxe, pick it up, and build on with exactly the blocks the blueprint needs.

Anomaly monitor (`RepairMonitor`; reads only the seen map, the in-hand counts and the blueprint instruction):
- `wrong_kind`: a blueprint cell holds a solid of another block than the blueprint names (multi-material blueprints);
- `must_be_empty`: a cell the blueprint marks 'air' holds a solid;
- `stray`: a cell outside the blueprint that the seen map measured as air holds one of the given blocks;
- `unaccounted` (ledger): for a given block, given - in hand > seen in cells that were air + owed pickups, for LEDGER_ROWS
  frames in a row (the server inventory lags placements by 4 rows): a block went where the sensors did not see. It is
  recorded; it has no behaviour of its own.
A block of the right kind in another blueprint cell is an out-of-order fill, not an anomaly. Every placement in the
85 sensor runs of 11-22/23 showed in the seen map on its press row, in a cell measured air the row before.

Dig skill (category block x attack, its own head): the nearest anomaly cell is the target. reach / move walk like the
building teacher toward a standpoint that sees a face of the block; dig (in reach, a face seen) turns to the face and
attacks only while the straight crosshair ray strikes it (a strike then holds anywhere on the block); collect (the dug
cell turned to air) walks to a floor cell within COLLECT_REACH of the drop, open blueprint cells included, until the
in-hand count of the dug block is one more than at the dig, at most COLLECT_ROWS frames (a drop no floor cell reaches,
e.g. on top of the beacon pyramid's centre, is then lost and the game cannot finish with exact gifts).
Entity rays (2026-11-23): a drop can slide off its cell. When a view ray has met a dropped item of the dug block since
the dig ('item:<block>', `pipeline.entity_track`), collect walks to that item's last seen cell and looks at it, and the
face features point at it; when none was seen it walks to the dug cell, and once there it looks around (turns while
looking down) until a ray meets the drop or the pickup counts. Records without entity rays replay as before.
The session's disclosed rule holds the pickaxe in reach / move / dig and the current cell's block while building.

`SensorRepairFeed` is the dig-only feed (teacher sensor_dig: the dig head's DIG_STAGES) and the beacon demo's composite
(teacher sensor_repair: the building head's stages while no anomaly is open, else the dig head's stages prefixed
'dig_'; COMBINED_STAGES). Success, filled cells and wrong placements come from the server placement and destroy records
and the server inventory (`RepairJudge`), never from the decision path.
"""
import dataclasses
import math
import random
from collections import Counter
from dataclasses import dataclass

from .build_recover import (ARRIVE, BUILD2_FEATURES, BUILD2_STAGES, INNER, MARGIN_EYES, SENSOR_DIG, SENSOR_REPAIR, STAND_REACH,
                            RecoverView, build2_features, look, teacher_action)
from .build_repair import struck_face, visible_block_face
from .build_sensor import SensorBuildTracker, blueprint_of
from .build_task import SCHEMA, build_fixture
from .pipeline.entity_track import EntityTracks
from .pipeline.seen_map import PASSABLE, UNKNOWN
from .pipeline.stage_replay import fixture_origin
from .place_task import EYE, FRONT, REACH, _at, _centre, _turn
from .reacquire_teacher import BUTTONS, point_delta

PICKAXE = 'diamond_pickaxe'
SUPPORT = 'stone_bricks'
DIG_KINDS = ('gold_block', 'iron_block', 'diamond_block', 'beacon')
DIG_STAGES = ('reach', 'move', 'dig', 'collect', 'done')
DIG_LABELLED = ('reach', 'move', 'dig', 'collect')
DIG_FEATURES = tuple('d_' + name for name in BUILD2_FEATURES)
DIG_PREFIX = 'dig_'
COMBINED_STAGES = BUILD2_STAGES + tuple(DIG_PREFIX + s for s in DIG_STAGES)
COLLECT_REACH = 1.      # blocks, horizontal cell distance: a floor cell this close picks the drop up (pickup box 1.3 + item)
COLLECT_ROWS = 100      # frames a pickup may take before the drop counts as lost
LEDGER_ROWS = 10        # frames a ledger shortfall must last before it is an anomaly
ITEM_HALF = .125        # blocks: half the width of a dropped item; a ray meets its near face, the centre is this far on
COLLECT_POINT_REACH = .8  # blocks, horizontal: a standpoint this close to a seen drop's centre picks it up
WALK_TO = 1.1           # blocks: an arrived collect still this far from a seen drop walks straight on toward it
SCAN_YAW = 20.          # degrees per frame an arrived collect turns while it sees no drop
SCAN_PITCH = 35.        # degrees below the horizon it looks while turning


def item_centre(point, eye, half=ITEM_HALF):
    """A dropped item's centre estimated from a ray's hit point on its near face: `half` further away from the eye,
    horizontally only (the item lies on a surface; its height stays the hit point's)."""
    dx, dz = point[0] - eye[0], point[2] - eye[2]; d = math.hypot(dx, dz)
    return tuple(point) if d == 0 else (point[0] + dx / d * half, point[1], point[2] + dz / d * half)


def blueprint_empty(blueprint):
    """The cells a blueprint_v1 marks 'air' (must stay empty), relative to the start block."""
    low = blueprint['min']; sx, sy, sz = blueprint['size']; blocks = blueprint['blocks']
    return [(x + low[0], y + low[1], z + low[2]) for x in range(sx) for y in range(sy) for z in range(sz)
            if blocks[x][y][z] == 'air']


def dig_task(seed):
    """Dig-only training task `seed`: 1 + seed % 3 wrong blocks of the beacon kinds on cells 2-6 ahead and -3..3
    sideways, no two next to each other (diagonals included), each on the floor or, one time in three, on a stone-brick
    support. The blueprint marks their cells 'air'; nothing is built."""
    rng = random.Random(seed * 7919 + 23)
    options = [(dx, dz) for dx in range(-3, 4) for dz in range(2, 7)]
    picks = []
    for _ in range(1 + seed % 3):
        free = [o for o in options if all(max(abs(o[0] - p[0]), abs(o[1] - p[1])) > 1 for p in picks)]
        picks.append(rng.choice(free))
    wrong, supports = [], []
    for dx, dz in picks:
        up = int(rng.random() < 1 / 3)
        if up: supports.append([dx, 0, dz])
        wrong.append([dx, up, dz, rng.choice(DIG_KINDS)])
    return dict(schema=SCHEMA, seed=seed, shape='dig%d' % len(wrong), cells=[], empty=[w[:3] for w in wrong], wrong=wrong,
                supports=supports, block=wrong[0][3])


def repair_fixture(task, teacher=SENSOR_REPAIR):
    """DemoFixture config of a dig task (teacher sensor_dig) or a building task with repair (sensor_repair): the block
    placing arena, the task's supports and wrong blocks set, and exact gifts: every blueprint block as many times as its
    cells need, in first-use order (a dig task: one of each beacon kind), then the pickaxe in the next hotbar slot."""
    if teacher not in (SENSOR_DIG, SENSOR_REPAIR): raise ValueError('teacher must be %r or %r' % (SENSOR_DIG, SENSOR_REPAIR))
    if (teacher == SENSOR_DIG) != (not task['cells']):
        raise ValueError('a dig task builds nothing; a repair task has cells to build')
    fixture = build_fixture(task)
    fixture['build'] = dict(task, teacher=teacher)
    kinds = [c[3] if len(c) > 3 else task['block'] for c in task['cells']] or list(DIG_KINDS)
    fixture['give'] = [dict(item=k, count=kinds.count(k)) for k in dict.fromkeys(kinds)] + [dict(item=PICKAXE, count=1)]
    fixture['blocks'] = ([dict(dx=s[0], dy=s[1], dz=s[2], block=SUPPORT) for s in task.get('supports', ())]
                         + [dict(dx=w[0], dy=w[1], dz=w[2], block=w[3]) for w in task.get('wrong', ())])
    fixture['description'] += (' Sensor repair (2026-11-23): %s; exact gifts %s and a %s; success needs every blueprint '
                               'cell of its block, the cells marked empty empty and no stray left (server records).'
                               % ('dig task, %d wrong blocks to dig and pick up' % len(task.get('wrong', ()))
                                  if teacher == SENSOR_DIG else 'building with repair',
                                  ', '.join('%s x%d' % (g['item'], g['count']) for g in fixture['give'][:-1]), PICKAXE))
    return fixture


def gifts(fixture):
    """Blocks handed at the start (instruction: what the task gives), the pickaxe left out."""
    return {g['item']: int(g['count']) for g in fixture.get('give', ()) if g['item'] != PICKAXE}


class RepairMonitor:
    """The anomaly rules of the module doc, run on every sensor frame."""

    def __init__(self, kind_of, empty, given, check_kinds):
        self.kind_of = dict(kind_of); self.empty = set(empty); self.given = dict(given); self.check_kinds = bool(check_kinds)
        self.measured_air = set(); self.flagged = {}; self.counts = Counter(); self.events = []
        self.short_rows = 0; self.unaccounted = 0; self.frame = 0

    def update(self, seen, inventory, owed=()):
        """The anomaly cells now, {cell: (rule, block)}; first flags and clears go to the events."""
        self.frame += 1; found = {}; ours = Counter()
        for cell, (kind, _, _) in seen.cells.items():
            if kind == 'air': self.measured_air.add(cell); continue
            if kind in PASSABLE or kind == UNKNOWN: continue
            if cell in self.measured_air: ours[kind] += 1
            if cell in self.kind_of:
                if self.check_kinds and kind != self.kind_of[cell]: found[cell] = ('wrong_kind', kind)
            elif cell in self.empty: found[cell] = ('must_be_empty', kind)
            elif cell in self.measured_air and kind in self.given: found[cell] = ('stray', kind)
        for cell, (rule, kind) in found.items():
            if self.flagged.get(cell) != (rule, kind):
                self.counts[rule] += 1
                expected = self.kind_of.get(cell, 'air' if cell in self.empty else None)
                self.events.append(dict(event='anomaly', rule=rule, cell=list(cell), block=kind, expected=expected,
                                        frame=self.frame))
        for cell in sorted(set(self.flagged) - set(found)):
            self.events.append(dict(event='anomaly_cleared', rule=self.flagged[cell][0], cell=list(cell), frame=self.frame))
        self.flagged = found
        owed = Counter(owed)
        short = sorted(k for k, n in self.given.items() if n - inventory.get(k, 0) > ours[k] + owed[k])
        self.short_rows = self.short_rows + 1 if short else 0
        if self.short_rows == LEDGER_ROWS:
            self.unaccounted += 1; self.events.append(dict(event='unaccounted', blocks=short, frame=self.frame))
        return found

    def pop_events(self):
        out, self.events = self.events, []
        return out


@dataclass(frozen=True)
class DigView(RecoverView):
    strike: bool = False     # dig: the straight crosshair ray strikes the target block
    seen_item: bool = False  # collect: a ray met the dug block's drop since the dig (target = its last seen cell)


class SensorDigTracker(SensorBuildTracker):
    """Stage machine of the dig skill on the seen map; `targets` ({cell: (rule, block)}) come from the monitor."""

    def __init__(self, blueprint, origin, yaw):
        super().__init__(blueprint, origin, yaw)
        self.targets = {}; self.digging = None; self.owed = None; self.striking = None
        self.dug = 0; self.collected = 0; self.lost = 0; self.events = []

    def owed_counts(self): return Counter() if self.owed is None else Counter({self.owed['block']: 1})

    def _usable_dig(self, cell, block):
        eye = (cell[0] + .5, float(cell[1]) + EYE, cell[2] + .5)
        if math.dist(eye, _centre(block)) > STAND_REACH: return False
        return all(visible_block_face((eye[0] + a, eye[1], eye[2] + b), block, self.solids) is not None
                   for a, b in MARGIN_EYES)

    def _usable_collect(self, cell, drop):
        return math.hypot(cell[0] - drop[0], cell[2] - drop[2]) <= COLLECT_REACH + 1e-9 and 0 <= drop[1] - cell[1] <= 1

    def update_dig(self, x, y, z, yaw, pitch, inventory, entities=None, frame=0):
        eye = (x, y + EYE, z); filled = len(self.filled_cells)
        if self.digging is not None and self.digging['cell'] not in self.solids:
            d = self.digging; self.digging = None; self.dug += 1
            self.owed = dict(cell=d['cell'], block=d['block'], need=inventory.get(d['block'], 0) + 1, frames=0, since=frame)
            self.events.append(dict(event='dug', cell=list(d['cell']), block=d['block']))
        left = len(self.targets) + (self.owed is not None)
        if self.owed is not None:
            o = self.owed
            if inventory.get(o['block'], 0) >= o['need']:
                self.owed = None; self.collected += 1
                self.events.append(dict(event='collected', cell=list(o['cell']), block=o['block'], frames=o['frames']))
            elif o['frames'] >= COLLECT_ROWS:
                self.owed = None; self.lost += 1
                self.events.append(dict(event='collect_lost', cell=list(o['cell']), block=o['block'], frames=o['frames']))
            else:
                o['frames'] += 1
                track = None if entities is None else entities.latest('item:' + o['block'], since=o['since'])
                cell, face, usable = o['cell'], None, self._usable_collect
                if track is not None:
                    point = track['point']; cell = tuple(math.floor(v) for v in point)
                    face = dict(block=None, normal=None, centre=point)

                    def usable(c, _, p=point, top=cell[1]):
                        return math.hypot(c[0] + .5 - p[0], c[2] + .5 - p[2]) <= COLLECT_POINT_REACH and 0 <= top - c[1] <= 1
                stand, way = self.standpoint(x, y, z, cell, usable=usable, stand_in_open=True)
                return DigView('collect', cell, face, math.dist(eye, _centre(cell)), filled, left, stand, way,
                               seen_item=track is not None)
        left = len(self.targets)
        if not self.targets:
            self.striking = None
            return DigView('done', None, None, 0., filled, 0, None, None)
        target = min(self.targets, key=lambda c: (math.dist(eye, _centre(c)), c))
        distance = math.dist(eye, _centre(target))
        margin = 0. if self.striking == target else INNER
        struck = struck_face(eye, look(yaw, pitch), self.solids, target, margin=margin)
        self.striking = target if struck is not None else None
        seen = struck or visible_block_face(eye, target, self.solids)
        if distance <= REACH and seen is not None:
            if self.digging is None or self.digging['cell'] != target:
                self.digging = dict(cell=target, block=self.targets[target][1])
            return DigView('dig', target, seen, distance, filled, left, None, None, strike=struck is not None)
        stand, way = self.standpoint(x, y, z, target, usable=self._usable_dig)
        return DigView('reach' if distance > REACH else 'move', target, seen, distance, filled, left, stand, way)


def dig_action(x, y, z, yaw, pitch, view):
    """MineRL-style teacher label of the dig skill, None when done or unknown (module doc)."""
    if view is None or view.stage not in DIG_LABELLED: return None
    eye = (x, y + EYE, z)
    if view.stage in ('reach', 'move'): return teacher_action(x, y, z, yaw, pitch, view)
    if view.stage == 'collect':
        if view.stand is not None and math.hypot(view.stand[0] - x, view.stand[2] - z) > ARRIVE:
            return teacher_action(x, y, z, yaw, pitch, dataclasses.replace(view, stage='move'))
        action = {k: 0 for k in BUTTONS}
        if view.seen_item:
            point = view.face['centre']; dpitch, dyaw = point_delta(eye, yaw, pitch, point)
            action['camera'] = _turn(dpitch, dyaw)
            # Arrived but the drop is still out of reach (it slid on): walk straight toward it.
            if math.hypot(point[0] - x, point[2] - z) > WALK_TO: action['forward'] = int(abs(dyaw) <= FRONT)
        else: action['camera'] = _turn(SCAN_PITCH - pitch, SCAN_YAW)       # look around for the drop
        return action
    action = {k: 0 for k in BUTTONS}
    if view.strike:
        action['attack'] = 1; action['camera'] = _turn(0., 0.)
        return action
    point = view.face['centre'] if view.face is not None else _centre(view.target)
    action['camera'] = _turn(*point_delta(eye, yaw, pitch, point))
    return action


def dig_features(x, y, z, yaw, pitch, view):
    """DIG_FEATURES: the building features with dig shown as aim (toward the seen face) and collect as move."""
    if view is None or view.stage not in DIG_LABELLED: return [0.] * len(DIG_FEATURES)
    shown = dataclasses.replace(view, stage={'dig': 'aim', 'collect': 'move'}.get(view.stage, view.stage))
    return build2_features(x, y, z, yaw, pitch, shown)


class RepairJudge:
    """Server-record judge of a dig or repair task (report and success only): the cells' blocks from the fixture's blocks,
    the placement record and the destroy record; the server inventory for pickups."""

    def __init__(self, fixture):
        self.task = fixture['build']; self.dig_only = not self.task['cells']; self.given = gifts(fixture)
        self.fixture_blocks = [(b['dx'], b['dy'], b['dz'], b['block']) for b in fixture.get('blocks', ())]
        self.state = None; self.count = None; self.destroys = None; self.unknown = 0; self.unknown_destroys = 0
        self.placed = 0; self.wrong_placed = 0; self.dug = 0; self.broken = 0; self.inventory = {}
        self.fixture_dug = Counter()

    def start_step(self, start_t): pass

    def observe(self, row):
        if self.state is None:
            start = fixture_origin([row])
            if start is None: return
            origin, yaw = start
            self.kind_of = {_at(origin, yaw, *c[:3]): (c[3] if len(c) > 3 else self.task['block']) for c in self.task['cells']}
            self.empty = {_at(origin, yaw, *c[:3]) for c in self.task.get('empty', ())}
            self.state = {_at(origin, yaw, dx, dy, dz): b for dx, dy, dz, b in self.fixture_blocks}
            self.fixture_cells = set(self.state)
        d = row.get('demo') or {}; p = row.get('place') or {}
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        self.inventory = dict(row.get('inventory') or {})
        if p.get('place_available') == 1:
            count = int(p.get('place_count', 0))
            if self.count is None: self.count = count
            elif count > self.count:
                self.unknown += count - self.count - 1; self.count = count
                cell = tuple(int(p['place_last_' + a]) for a in 'xyz'); block = p.get('place_last_block')
                self.state[cell] = block; self.placed += 1
                self.wrong_placed += self.kind_of.get(cell) != block
        if d.get('demo_destroys') is not None:
            count = int(d['demo_destroys'])
            if self.destroys is None: self.destroys = count
            elif count > self.destroys:
                self.unknown_destroys += count - self.destroys - 1; self.destroys = count
                if d.get('demo_last_destroy_removed') == 1:
                    cell = tuple(int(d['demo_last_destroy_' + a]) for a in 'xyz')
                    before = self.state.pop(cell, None)
                    if cell in self.kind_of and before == self.kind_of[cell]: self.broken += 1
                    elif cell in self.empty or (before is not None and cell not in self.fixture_cells): self.dug += 1
                    if cell in self.fixture_cells and cell in self.empty: self.fixture_dug[before] += 1

    def wrong_cells(self):
        return [] if self.state is None else sorted(c for c, k in self.kind_of.items() if c in self.state and self.state[c] != k)

    def strays(self):
        """Blocks standing where nothing may stand: a cell marked empty, or one of our placements outside the blueprint."""
        if self.state is None: return []
        return sorted(c for c in self.state if c not in self.kind_of and (c in self.empty or c not in self.fixture_cells))

    def done(self):
        if self.state is None or any(self.state.get(c) != k for c, k in self.kind_of.items()) or self.strays(): return False
        if self.dig_only:
            return all(self.inventory.get(k, 0) >= self.given.get(k, 0) + n for k, n in self.fixture_dug.items())
        return True

    def filled(self):
        return 0 if self.state is None else sum(1 for c, k in self.kind_of.items() if self.state.get(c) == k)

    def wrong(self): return self.wrong_placed


class SensorRepairFeed:
    """Live and replayed sensor repair feed (module doc). Decisions come from sensor frames, the seen map and the
    blueprint (`observe_frame`); `observe(row)` only feeds the judge."""

    def __init__(self, fixture):
        self.task = fixture['build']; self.combined = self.task.get('teacher') == SENSOR_REPAIR
        self.blueprint = blueprint_of(self.task); self.given = gifts(fixture)
        self.judge = RepairJudge(fixture)
        self.build = self.dig = self.monitor = None; self.mode = None
        self.entities = EntityTracks(); self.frames = 0
        self.last_view = self.last_local = self.last_frame = self.last_row = None; self.events = []

    def start_step(self, start_t): self.start_t = start_t; self.judge.start_step(start_t)

    def observe(self, row): self.last_row = row; self.judge.observe(row)

    def observe_frame(self, frame, seen):
        if frame is None: return
        if self.build is None:
            self.build = SensorBuildTracker(self.blueprint, frame.foot, frame.yaw)
            self.dig = SensorDigTracker(self.blueprint, frame.foot, frame.yaw)
            empty = [_at(frame.foot, frame.yaw, *c) for c in blueprint_empty(self.blueprint)]
            self.monitor = RepairMonitor(self.build.kind_of, empty, self.given, self.build.check_kinds)
        self.build.sync(seen); self.dig.sync(seen); self.last_frame = frame
        self.frames += 1
        self.entities.update(self.frames, [(k, item_centre(p, frame.eye) if k.startswith('item:') else p, d)
                                           for k, p, d in frame.entity_sightings()])
        self.dig.targets = self.monitor.update(seen, frame.inventory, owed=self.dig.owed_counts())
        x, y, z = frame.position
        view = self.dig.update_dig(x, y, z, frame.yaw, frame.pitch, frame.inventory, self.entities, self.frames)
        self.events += self.monitor.pop_events() + self.dig.events; self.dig.events = []
        mode = self.mode
        if view.stage != 'done' or not self.combined:
            self.mode = 'dig'; self.last_local = view
            self.last_view = dataclasses.replace(view, stage=DIG_PREFIX + view.stage) if self.combined else view
        else:
            kind, cell = frame.crosshair
            hit = cell if kind != 'air' else None
            face = frame.hit_face if hit is not None else None
            self.mode = 'build'; self.last_local = self.last_view = self.build.update(x, y, z, frame.yaw, frame.pitch, hit, face)
        if self.combined and mode is not None and mode != self.mode:
            self.events.append(dict(event='head', head=self.mode, frame=self.monitor.frame))

    def decision_view(self): return self.last_view

    def pop_events(self):
        out, self.events = self.events, []
        return out

    def needed_kind(self):
        """What the session's rule puts in hand: the pickaxe while digging (reach, move, dig), nothing while collecting,
        the current cell's block while building; None before the first frame or when nothing is left."""
        if self.mode is None: return None
        if self.mode == 'dig': return PICKAXE if self.last_local.stage in ('reach', 'move', 'dig') else None
        cell = self.build.current()
        return None if cell is None else self.build.kind_of[cell]

    def done(self): return self.judge.done()

    def filled(self): return self.judge.filled()

    def wrong(self): return self.judge.wrong()

    @property
    def unknown(self): return self.judge.unknown

    def label(self, row, view):
        f = self.last_frame
        if f is None or view is None or self.last_local is None: return None
        act = dig_action if self.mode == 'dig' else teacher_action
        return act(*f.position, f.yaw, f.pitch, self.last_local)

    def features(self):
        f = self.last_frame
        if f is None or self.last_local is None: return [0.] * len(DIG_FEATURES)
        return (dig_features if self.mode == 'dig' else build2_features)(*f.position, f.yaw, f.pitch, self.last_local)

    def repair_extra(self):
        j, m, d = self.judge, self.monitor, self.dig
        return dict(dug=j.dug, broken=j.broken, strays_left=len(j.strays()), wrong_left=len(j.wrong_cells()),
                    unknown_destroys=j.unknown_destroys, anomalies=dict(m.counts) if m else {},
                    unaccounted=m.unaccounted if m else 0, repairs=d.dug if d else 0, collected=d.collected if d else 0,
                    collect_lost=d.lost if d else 0)
