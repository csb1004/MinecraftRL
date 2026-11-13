"""Bridging skill (design docs/specs/2026-11-14-bridge-design.md): fill a 3-5 block gap between a start platform and a
gold goal platform, then step onto the goal.

Task coordinates are relative to the start block (dx right, dy up, dz forward of the start facing) like DemoFixture.
The bridge line is the start column (dx 0) one below the start block (the platforms' block row). The teacher turns its
back to the gap, sneaks backward to the edge (sneaking stops the player 0.25-0.3 past it), aims at the face of the last
block that points into the gap and places; it repeats until the line reaches the goal, then backs onto the goal.

Stages and labels read the client pose, the client crosshair and the task (privileged); success and falls read the
server pose, and the filled cells read the server placement record (opt-in place monitor) only.
"""
import math
import random
from dataclasses import dataclass

import numpy as np

from .jump_course import FALL_MARGIN, GROUND_EPS, _pose, world_platforms
from .pipeline.stage_replay import FACINGS, facing_index, fixture_origin
from .place_task import FACE_NORMALS, GIVE_COUNT, _at, _hit
from .reacquire_teacher import BUTTONS, point_delta

SCHEMA = 'bridge_task_v1'
BRIDGE_STAGES = ('turn', 'walk', 'aim', 'place', 'fallen', 'goal')
LABELLED = ('turn', 'walk', 'aim', 'place')
SNEAK_EYE = 1.27       # eye height while sneaking
FRONT = 20.            # degrees of yaw off the walk heading that call for turning first
AIM_PITCH = 80.        # degrees down while turning and walking (the face needs about 77-83 at the edge)
EDGE_AIM = .15         # blocks the centre is past the edge before aiming
LOOK_AHEAD = 2.        # blocks along the line: the walk heading faces away from this point (back onto the line)
MAX_TURN = 10.         # degrees per axis per decision
GAPS = (3, 4, 5)
GOAL_SHIFT = (-1, 0, 1)
START = dict(x0=-1, x1=1, z0=-1, z1=2, top=0)
GOAL_LENGTH = 2
SIDE = 4               # the arena spans dx -SIDE..SIDE
BRIDGE_FEATURES = ('h_sin', 'h_cos', 'a_pitch', 'edge', 'left')
FEATURE_PITCH = 60.
EDGE_SCALE = 3.
LEFT_SCALE = 5.


def generate_bridge(seed):
    """A task from `seed`: the gap length (3-5) and the goal platform's sideways shift (-1..1)."""
    rng = random.Random(seed)
    return dict(schema=SCHEMA, seed=seed, gap=rng.choice(GAPS), goal_dx=rng.choice(GOAL_SHIFT), block='cobblestone')


def platforms(task):
    """The start and the goal platform in the jump course format (block bounds of the upper face rows, `top`)."""
    start = dict(START, gap=0, block='stone_bricks')
    z0 = START['z1'] + task['gap'] + 1
    goal = dict(x0=task['goal_dx'] - 1, x1=task['goal_dx'] + 1, z0=z0, z1=z0 + GOAL_LENGTH - 1, top=START['top'],
                gap=task['gap'], block='gold_block')
    return [start, goal]


def bridge_cells(task):
    """Relative cells of the bridge line, from the start edge to the goal: [dx, dy, dz]."""
    return [[0, START['top'] - 1, START['z1'] + 1 + k] for k in range(task['gap'])]


def bridge_fixture(task):
    """DemoFixture config: stone and a grass floor (top 3 below the platforms), cleared air, barrier walls, the two
    platforms, and the blocks to place. The task rides along under `bridge` (DemoFixture ignores it)."""
    plats = platforms(task); end = plats[1]['z1']; w = SIDE
    fill = [dict({'from': [-w, -8, -3], 'to': [w, -5, end]}, block='stone'),
            dict({'from': [-w, -4, -3], 'to': [w, -4, end]}, block='grass_block'),
            dict({'from': [-w, -3, -3], 'to': [w, 8, end]}, block='air'),
            dict({'from': [-w - 1, -8, -4], 'to': [-w - 1, 8, end + 1]}, block='barrier'),
            dict({'from': [w + 1, -8, -4], 'to': [w + 1, 8, end + 1]}, block='barrier'),
            dict({'from': [-w, -8, -4], 'to': [w, 8, -4]}, block='barrier'),
            dict({'from': [-w, -8, end + 1], 'to': [w, 8, end + 1]}, block='barrier'),
            dict({'from': [-w, -9, -3], 'to': [w, -9, end]}, block='barrier')]
    blocks = [dict(dx=x, dy=p['top'] - 1, dz=z, block=p['block']) for p in plats
              for x in range(p['x0'], p['x1'] + 1) for z in range(p['z0'], p['z1'] + 1)]
    return dict(schema='iron_demo_fixture_v1', rotate_with_facing=True, fill=fill, blocks=blocks,
                give=[dict(item=task['block'], count=GIVE_COUNT)], bridge=task,
                description='Bridging task %s (design 2026-11-14): fill a %d block gap to the gold platform and step on it.'
                            % (task['seed'], task['gap']))


def world_bridge(task, origin, yaw):
    """World geometry of a task applied at `origin` (start block) facing `yaw`: the bridge cells in order, the goal
    platform's upper face, the platform top, and the line (start block centre, forward and right unit vectors, the
    along-line coordinate of the start edge)."""
    i = facing_index(yaw)
    (fx, fz), (rx, rz) = FACINGS[i], FACINGS[(i + 1) % 4]
    ox, oy, oz = origin
    goal = world_platforms(dict(platforms=platforms(task)), origin, yaw)[1]
    return dict(cells=[_at(origin, yaw, *c) for c in bridge_cells(task)], goal=goal, top=float(oy + START['top']),
                line=dict(x=ox + .5, z=oz + .5, ux=fx, uz=fz, rx=rx, rz=rz, edge=START['z1'] + .5))


def along(world, x, z):
    """Blocks along the bridge direction from the start block centre."""
    ln = world['line']
    return (x - ln['x']) * ln['ux'] + (z - ln['z']) * ln['uz']


def wrap(angle):
    return (angle + 180.) % 360. - 180.


def walk_heading(world, x, z):
    """Yaw that turns the back to the line point LOOK_AHEAD blocks further along: walking backward then heads there."""
    ln = world['line']; s = along(world, x, z) + LOOK_AHEAD
    bx, bz = ln['x'] + ln['ux'] * s - x, ln['z'] + ln['uz'] * s - z
    return math.degrees(math.atan2(bx, -bz))


@dataclass(frozen=True)
class BridgeView:
    stage: str
    filled: int            # bridge cells filled (server placement record)
    nxt: tuple | None      # the next cell to fill (world), None once the line reaches the goal
    face: tuple | None     # centre of the face to click for the next cell
    heading: float         # walk heading (yaw)
    past: float            # blocks the centre is past the current edge
    left: int              # cells still open


class BridgeTracker:
    """Skill-internal stage machine of one bridging run (design §2)."""

    def __init__(self, world):
        self.world = world; self.cells = [tuple(c) for c in world['cells']]
        self.placed_cells = set(); self.wrong = 0
        self.low = world['top'] - FALL_MARGIN

    def placed(self, cell):
        """A server placement (from the place record)."""
        cell = tuple(int(c) for c in cell)
        if cell in self.cells: self.placed_cells.add(cell)
        else: self.wrong += 1

    def filled(self):
        n = 0
        while n < len(self.cells) and self.cells[n] in self.placed_cells: n += 1
        return n

    def on_goal(self, x, y, z, on_ground):
        g = self.world['goal']
        return bool(on_ground) and abs(y - g['top']) <= GROUND_EPS and g['X0'] <= x <= g['X1'] and g['Z0'] <= z <= g['Z1']

    def update(self, x, y, z, yaw, pitch, on_ground, hit=None, face=None):
        """`hit`: the crosshair block (world cell) or None; `face`: its face ordinal (FACE_NORMALS)."""
        ln = self.world['line']; n = self.filled()
        nxt = self.cells[n] if n < len(self.cells) else None
        edge = ln['edge'] + n
        centre = None if nxt is None else (ln['x'] + ln['ux'] * edge, self.world['top'] - .5, ln['z'] + ln['uz'] * edge)
        past = along(self.world, x, z) - edge
        heading = walk_heading(self.world, x, z)

        def view(stage): return BridgeView(stage, n, nxt, centre, heading, past, len(self.cells) - n)

        if y < self.low: return view('fallen')
        if self.on_goal(x, y, z, on_ground): return view('goal')
        if nxt is not None and on_ground and past >= EDGE_AIM:
            if hit is not None and face is not None and 0 <= int(face) < 6:
                lands = tuple(h + k for h, k in zip(hit, FACE_NORMALS[int(face)]))
                if lands == nxt: return view('place')
            return view('aim')
        return view('turn' if abs(wrap(heading - yaw)) > FRONT else 'walk')


def _turn(dpitch, dyaw):
    return np.array([max(-MAX_TURN, min(MAX_TURN, dpitch)), max(-MAX_TURN, min(MAX_TURN, dyaw))])


def _look(x, y, z, yaw, pitch, view):
    """(pitch, yaw) differences to where the teacher looks: the face centre at the edge, else the walk heading at
    AIM_PITCH down."""
    if view.stage in ('aim', 'place') and view.face is not None:
        return point_delta((x, y + SNEAK_EYE, z), yaw, pitch, view.face)
    return AIM_PITCH - pitch, wrap(view.heading - yaw)


def teacher_action(x, y, z, yaw, pitch, view):
    """MineRL-style label for one state (design §3), or None on terminal stages."""
    if view is None or view.stage not in LABELLED: return None
    action = {k: 0 for k in BUTTONS}
    action['sneak'] = 1
    action['camera'] = _turn(*_look(x, y, z, yaw, pitch, view))
    if view.stage == 'walk': action['back'] = 1
    if view.stage == 'place': action['use'] = 1
    return action


def bridge_features(x, y, z, yaw, pitch, view):
    """Privileged head input next to the stage, in BRIDGE_FEATURES order: sin, cos of the yaw and pitch / 60 to where the
    teacher looks, the blocks still to go to the edge / 3 (negative once past it, -1/3..1), and the open cells / 5.
    Zeros on terminal or unknown stages."""
    if view is None or view.stage not in LABELLED: return [0.] * len(BRIDGE_FEATURES)
    dpitch, dyaw = _look(x, y, z, yaw, pitch, view)
    return [math.sin(math.radians(dyaw)), math.cos(math.radians(dyaw)), max(-1., min(1., dpitch / FEATURE_PITCH)),
            max(-1. / 3, min(1., -view.past / EDGE_SCALE)), view.left / LEFT_SCALE]


class BridgeFeed:
    """Live (and replayed) bridging stages from recorded telemetry rows, as JumpFeed and PlaceFeed do.

    Decision views (head input and teacher labels) follow the client pose and crosshair; server views decide success
    (goal) and falls. Both trackers take the filled cells from the server placement record; a count already present on
    the first row belongs to earlier steps."""

    def __init__(self, fixture):
        self.task = fixture['bridge']; self.world = None; self.tracker = None; self.server = None; self.count = None
        self.last_view = None; self.last_server_view = None; self.last_row = None; self.unknown = 0

    def start_step(self, start_t): self.start_t = start_t

    def observe(self, row):
        self.last_row = row; self.last_view = None; self.last_server_view = None
        if self.world is None:
            start = fixture_origin([row])
            if start is None: return
            self.world = world_bridge(self.task, *start)
            self.tracker = BridgeTracker(self.world); self.server = BridgeTracker(self.world)
        d = row.get('demo') or {}; p = row.get('place') or {}
        if not (row.get('server_available') == 1 and d.get('demo_available') == 1): return
        if p.get('place_available') == 1:
            count = int(p.get('place_count', 0))
            if self.count is None: self.count = count
            elif count > self.count:
                self.unknown += count - self.count - 1        # placements between two rows: only the last one is known
                self.count = count
                cell = tuple(int(p['place_last_' + a]) for a in 'xyz')
                self.tracker.placed(cell); self.server.placed(cell)
        try: yaw, pitch = float(row['yaw']), float(row['pitch'])
        except (KeyError, TypeError, ValueError): return
        server = _pose(row, 'server')
        if server is not None:
            x, y, z, ground, _ = server
            self.last_server_view = self.server.update(x, y, z, yaw, pitch, ground)
        client = _pose(row, 'client')
        if client is not None:
            x, y, z, ground, _ = client
            hit, face = _hit(row)
            self.last_view = self.tracker.update(x, y, z, yaw, pitch, ground, hit, face)

    def decision_view(self): return self.last_view

    def server_view(self): return self.last_server_view

    def filled(self): return 0 if self.server is None else self.server.filled()

    def wrong(self): return 0 if self.server is None else self.server.wrong

    def label(self, row, view):
        pose = None if row is None else _pose(row, 'client')
        if pose is None: return None
        x, y, z, _, _ = pose
        return teacher_action(x, y, z, float(row['yaw']), float(row['pitch']), view)

    def features(self):
        pose = None if self.last_row is None else _pose(self.last_row, 'client')
        if pose is None or self.last_view is None: return [0.] * len(BRIDGE_FEATURES)
        x, y, z, _, _ = pose
        return bridge_features(x, y, z, float(self.last_row['yaw']), float(self.last_row['pitch']), self.last_view)
