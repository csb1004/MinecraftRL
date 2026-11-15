"""Building round 5: removing stray blocks (design docs/specs/2026-11-15-build-repair-design.md).

The careful building skill (build_recover, round 4) plus a `dig` stage. A stray block is a block outside the blueprint:
one the fixture set (repair tasks) or one the server placement record put outside the blueprint (a wrong placement).
While a stray is left, the nearest one is the target: `dig` when it is in reach and one of its own faces is seen (or the
straight crosshair ray already meets it well inside a face), else reach / move to a standpoint that sees one of its faces.
The teacher switches to the stone pickaxe (hotbar slot 2) in `dig` and attacks without turning only on such a strike;
in the other stages it switches back to the block (slot 1) before clicking. Slot switches are actions, not stages.
The server destroy record removes blocks: a removed stray is dug, a removed blueprint block opens its cell again.
Done (success) is the blueprint filled and no stray left. The hand is the server's selected item (also on the hotbar).
"""
import dataclasses
import math
import random
from dataclasses import dataclass

from .build_recover import (BUILD2_FEATURES, CAREFUL, INNER, MARGIN_EYES, REPAIR, STAND_REACH, CarefulFeed, CarefulTracker,
                            RecoverView, _ordinal, build2_features, look, raycast, world_build2)
from .build_recover import teacher_action as careful_action
from .build_task import build_fixture, generate_build
from .pipeline.stage_replay import fixture_origin
from .place_task import ARENA, EYE, FACE_NORMALS, GIVE_COUNT, REACH, _at, _centre, _client, _hit, _turn
from .reacquire_teacher import BUTTONS, point_delta

REPAIR_STAGES = ('reach', 'move', 'aim', 'place', 'dig', 'done')
REPAIR_LABELLED = ('reach', 'move', 'aim', 'place', 'dig')
REPAIR_FEATURES = BUILD2_FEATURES + ('held',)
PICKAXE = 'stone_pickaxe'
BLOCK_SLOT, PICK_SLOT = 'hotbar.1', 'hotbar.2'     # the fixture gives the block first, then the pickaxe
STRIKE_REACH = 4.5                                   # blocks: survival reach of an attack from the eye
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def repair_build(seed):
    """build_task.generate_build(seed) plus `seed % 3` stray blocks: floor-level cells next to a blueprint floor cell,
    not a blueprint cell, not the start, inside the arena (design §2)."""
    task = generate_build(seed)
    blueprint = {tuple(c) for c in task['cells']}
    options = sorted({(x + dx, 0, z + dz) for x, y, z in blueprint if y == 0 for dx, dz in NEIGHBOURS
                      if (x + dx, 0, z + dz) not in blueprint and (x + dx, z + dz) != (0, 0)
                      and ARENA['x'][0] <= x + dx <= ARENA['x'][1] and ARENA['z'][0] <= z + dz <= ARENA['z'][1]})
    strays = random.Random(seed * 7919 + 5).sample(options, seed % 3)
    return dict(task, strays=[list(s) for s in strays])


def repair_fixture(task, teacher=REPAIR):
    """DemoFixture config of a repair task: the strays set as blocks, the block (slot 1) and a stone pickaxe (slot 2)
    given. teacher='careful' is the round 4 skill on the same world (the strays are fixed blocks to it)."""
    strays = [list(s) for s in task.get('strays', ())]
    if teacher == REPAIR:
        fixture = build_fixture(task)
        fixture['build'] = dict(task, teacher=REPAIR)
        fixture['description'] += ' Recovering teacher with careful place and stray removal (design 2026-11-15 build-repair).'
        ending = 'success also needs every stray removed'
    elif teacher == CAREFUL:
        fixture = build_fixture(task)
        fixture['build'] = dict(task, clutter=strays, teacher=CAREFUL)
        fixture['description'] += ' Recovering teacher with careful place (round 4), which does not remove stray blocks.'
        ending = 'the game ends when the blueprint is filled; stray removal is scored from the records'
    else: raise ValueError('teacher must be %r or %r' % (REPAIR, CAREFUL))
    fixture['blocks'] = [dict(dx=s[0], dy=s[1], dz=s[2], block=task['block']) for s in strays]
    fixture['give'] = [dict(item=task['block'], count=GIVE_COUNT), dict(item=PICKAXE, count=1)]
    fixture['description'] += ' Repair task: %d stray blocks set by the fixture, a %s in slot 2; %s.' % (
        len(strays), PICKAXE, ending)
    return fixture


def world_repair(task, origin, yaw):
    """build_recover.world_build2 plus the task's strays (solid) and its block name."""
    world = world_build2(task, origin, yaw)
    strays = [_at(origin, yaw, *s) for s in task.get('strays', ())]
    world['solids'] = set(world['solids']) | set(strays)
    return dict(world, strays=strays, block=task['block'])


def visible_block_face(eye, block, solids):
    """The face of the solid `block` nearest to the eye whose centre the eye's straight line reaches first (the face is
    open: no solid next to it, and the eye is on its outer side); None when no face is seen."""
    best = None
    block = tuple(block)
    for n in FACE_NORMALS:
        if tuple(b + k for b, k in zip(block, n)) in solids: continue
        centre = tuple(c + .5 * k for c, k in zip(_centre(block), n))
        if sum((e - c) * k for e, c, k in zip(eye, centre, n)) <= 0: continue
        d = math.dist(eye, centre)
        if best is not None and d >= best[0]: continue
        direction = tuple((c - e) / d for c, e in zip(centre, eye))
        if raycast(eye, direction, solids, reach=d + .01) != (block, _ordinal(n)): continue
        best = (d, dict(block=block, normal=tuple(n), centre=centre))
    return None if best is None else best[1]


def struck_face(eye, direction, solids, block, margin=INNER, reach=STRIKE_REACH):
    """The face of `block` the straight ray from the eye meets first, at least `margin` inside its edges, as
    dict(block, normal, centre); None when the ray meets another block first, misses, or lands near an edge."""
    cell, face = raycast(eye, direction, solids, reach=reach)
    if cell is None or face is None or cell != tuple(block): return None
    n = FACE_NORMALS[face]
    axis = next(i for i in range(3) if n[i])
    if not direction[axis]: return None
    plane = cell[axis] + (1 if n[axis] > 0 else 0)
    t = (plane - eye[axis]) / direction[axis]
    point = [e + t * d for e, d in zip(eye, direction)]
    if not all(margin <= point[i] - cell[i] <= 1 - margin for i in range(3) if i != axis): return None
    return dict(block=cell, normal=tuple(n), centre=tuple(c + .5 * k for c, k in zip(_centre(cell), n)))


@dataclass(frozen=True)
class RepairView(RecoverView):
    pick: bool = False       # the stone pickaxe is in hand
    block: bool = True       # the building block is in hand
    strike: bool = False     # dig: the straight crosshair ray meets the stray well inside a face
    stray: bool = False      # the target is a stray block (dig, or reach / move toward one)


class RepairTracker(CarefulTracker):
    """Skill-internal stage machine of one repair run (design §3). `held` is the item in hand, set before update."""

    def __init__(self, world):
        super().__init__(world)
        self.strays = {tuple(s) for s in world.get('strays', ())}
        self.block_item = world.get('block', 'cobblestone')
        self.dug = 0; self.broken = 0; self.held = None
        self.striking = None       # the stray struck at the last update (a strike then holds anywhere on its faces)

    def placed(self, cell):
        cell = tuple(int(c) for c in cell)
        super().placed(cell)
        if cell not in self.cells: self.strays.add(cell)

    def destroyed(self, cell):
        """A server destroy (from the destroy record): a stray is dug, a blueprint block opens its cell again."""
        cell = tuple(int(c) for c in cell); self.solids.discard(cell)
        if cell in self.strays: self.strays.discard(cell); self.dug += 1
        elif cell in self.filled_cells: self.filled_cells.discard(cell); self.broken += 1

    @property
    def done(self): return self.current() is None and not self.strays

    def _usable_dig(self, cell, block):
        eye = (cell[0] + .5, float(cell[1]) + EYE, cell[2] + .5)
        if math.dist(eye, _centre(block)) > STAND_REACH: return False
        return all(visible_block_face((eye[0] + a, eye[1], eye[2] + b), block, self.solids) is not None
                   for a, b in MARGIN_EYES)

    def update(self, x, y, z, yaw, pitch, hit, face):
        # An unknown hand (a record without the selected item) counts as the block: no switch label from missing data.
        hand = dict(pick=self.held == PICKAXE, block=self.held in (None, self.block_item))
        if not self.strays:
            self.striking = None
            view = super().update(x, y, z, yaw, pitch, hit, face)
            return RepairView(*(getattr(view, f.name) for f in dataclasses.fields(RecoverView)), **hand)
        eye = (x, y + EYE, z)
        target = min(self.strays, key=lambda s: (math.dist(eye, _centre(s)), s))
        filled = len(self.filled_cells); left = len(self.cells) - filled
        distance = math.dist(eye, _centre(target))
        # A strike starts well inside a face (INNER) and then holds while the ray stays anywhere on the stray: releasing
        # the attack resets the breaking progress in Minecraft, so the teacher must not stop at a face's rim.
        margin = 0. if self.striking == target and hand['pick'] else INNER
        struck = struck_face(eye, look(yaw, pitch), self.solids, target, margin=margin)
        self.striking = target if struck is not None and hand['pick'] else None
        seen = struck or visible_block_face(eye, target, self.solids)
        if distance <= REACH and seen is not None:
            return RepairView('dig', target, seen, distance, filled, left, None, None, strike=struck is not None,
                              stray=True, **hand)
        stand, way = self.standpoint(x, y, z, target, usable=self._usable_dig)
        return RepairView('reach' if distance > REACH else 'move', target, seen, distance, filled, left, stand, way,
                          stray=True, **hand)


def teacher_action(x, y, z, yaw, pitch, view):
    """MineRL-style label for one state, or None when done or unknown (design §3)."""
    if view is None or view.stage not in REPAIR_LABELLED: return None
    if view.stage != 'dig':
        action = careful_action(x, y, z, yaw, pitch, view)
        if action is not None and not view.stray and not view.block:
            action[BLOCK_SLOT] = 1; action['use'] = 0           # back to the block before any click
        return action
    action = {k: 0 for k in BUTTONS}
    if view.pick and view.strike:
        action['attack'] = 1; action['camera'] = _turn(0., 0.)
        return action
    point = view.face['centre'] if view.face is not None else _centre(view.target)
    action['camera'] = _turn(*point_delta((x, y + EYE, z), yaw, pitch, point))
    if not view.pick: action[PICK_SLOT] = 1
    return action


def repair_features(x, y, z, yaw, pitch, view):
    """REPAIR_FEATURES: the round 4 features (in dig: toward the stray and its seen face) and the pickaxe in hand."""
    if view is None or view.stage not in REPAIR_LABELLED: return [0.] * len(REPAIR_FEATURES)
    shown = dataclasses.replace(view, stage='aim') if view.stage == 'dig' else view
    return build2_features(x, y, z, yaw, pitch, shown) + [1. if view.pick else 0.]


class RepairFeed(CarefulFeed):
    """CarefulFeed with the repair tracker, teacher and features; it also follows the server destroy record (a count on
    the first row is a baseline; skipped destroys are counted) and the selected item."""
    TRACKER = RepairTracker

    def __init__(self, fixture):
        super().__init__(fixture); self.destroys = None; self.unknown_destroys = 0

    def observe(self, row):
        self.last_row = row; self.last_view = None
        if self.tracker is None:
            start = fixture_origin([row])
            if start is None: return
            self.tracker = self.TRACKER(world_repair(self.task, *start))
        d = row.get('demo') or {}; p = row.get('place') or {}
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        if p.get('place_available') == 1:
            count = int(p.get('place_count', 0))
            if self.count is None: self.count = count
            elif count > self.count:
                self.unknown += count - self.count - 1
                self.count = count
                self.tracker.placed(tuple(int(p['place_last_' + a]) for a in 'xyz'))
        if d.get('demo_destroys') is not None:
            count = int(d['demo_destroys'])
            if self.destroys is None: self.destroys = count
            elif count > self.destroys:
                self.unknown_destroys += count - self.destroys - 1
                self.destroys = count
                if d.get('demo_last_destroy_removed') == 1:
                    self.tracker.destroyed(tuple(int(d['demo_last_destroy_' + a]) for a in 'xyz'))
        self.tracker.held = row.get('server_selected_item')
        pose = _client(row)
        if pose is None: return
        hit, face = _hit(row)
        self.last_view = self.tracker.update(*pose, hit, face)

    def label(self, row, view):
        pose = None if row is None else _client(row)
        return None if pose is None else teacher_action(*pose, view)

    def features(self):
        pose = None if self.last_row is None else _client(self.last_row)
        if pose is None or self.last_view is None: return [0.] * len(REPAIR_FEATURES)
        return repair_features(*pose, self.last_view)

    def repair_extra(self):
        tr = self.tracker
        return dict(strays_left=len(self.task.get('strays', ())) if tr is None else len(tr.strays),
                    dug=0 if tr is None else tr.dug, broken=0 if tr is None else tr.broken,
                    unknown_destroys=self.unknown_destroys)


def repair_outcome(task, rows):
    """The repair rules replayed over any run's telemetry rows on `task` (a repair task with strays): blueprint filled
    and no stray left (`done`), with the counts. Used to score arms whose game did not use the repair feed."""
    feed = RepairFeed(dict(build=task))
    for r in rows: feed.observe(r)
    return dict(done=feed.done(), filled=feed.filled(), wrong=feed.wrong(), unknown_placements=feed.unknown,
                **feed.repair_extra())
