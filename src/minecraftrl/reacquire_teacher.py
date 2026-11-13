"""Privileged turn-to-target teacher (design 2026-11-07 §4): labels only, never dispatched during play.

Uses the server pose and the fixture's target positions (privileged observations). For a policy row in the near,
aimed or mining stage the label turns the view toward the target block centre by at most MAX_TURN degrees per
axis, attacks only while the crosshair is on the target family, and presses nothing else.

aim='approach' (design 2026-11-09 §3) adds walking: forward while the target is farther than CLOSE blocks
horizontally (and, when turning, roughly in front), and a collect label that turns toward the last broken target
block (server destroy position) and walks onto its drop.

aim='press' (design 2026-11-10 §3) is 'approach' with one label for the whole mining stage: attack with the view
still, also on the rows where the server still reports mining the target after the crosshair slipped off it. The
approach teacher let go and turned on those rows (15-17% of the mining stage), and the student, which cannot tell
them apart by its stage input, lowered its attack probability and broke its own mining.

aim='sight' (line-of-sight design 2026-11-10 §3) is 'press' with a near label that knows when the target is hidden.
The press teacher turned toward the block centre and stood still within CLOSE blocks even when another block was in
the way, and students stalled there. With the fixture geometry of the moment (broken blocks removed) the sight
teacher turns toward the centre only when a straight line from the eye reaches it; otherwise toward an open face
that is in sight; otherwise it walks to the spot STAND blocks in front of an open side face.
"""
import math

import numpy as np

from .pipeline.steps import block_matches

LABELLED_STAGES = ('near', 'aimed', 'mining')
APPROACH_STAGES = LABELLED_STAGES + ('collect',)
MAX_TURN = 10.
CLOSE = 2.          # blocks, feet to the target block centre, horizontal: mine from here or closer
DROP_CLOSE = .5     # blocks, feet to the broken block centre, horizontal: stop walking onto the drop
FRONT = 45.         # degrees of yaw: walk while turning only toward a target this close to straight ahead
INSET = .05         # blocks: the aim point of an open face lies this far inside the target
STAND = 1.5         # blocks from an open side face of the target to the standing spot in front of it
SPOT_CLOSE = .3     # blocks, feet to the standing spot, horizontal: stop walking
NORMALS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# steve1.VPT.lib.actions.Buttons.ALL (a test checks they match).
BUTTONS = ('attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 'drop', 'inventory',
           *('hotbar.%d' % i for i in range(1, 10)))


def point_delta(eye, yaw, pitch, point):
    """(dpitch, dyaw) in degrees from the current view to `point`.

    Same convention as stage_replay.view_angle: yaw 0 = +z, 90 = -x, pitch down positive. camera[0] adds to the
    pitch and camera[1] to the yaw. The yaw difference is wrapped to (-180, 180]."""
    to = (point[0] - eye[0], point[1] - eye[1], point[2] - eye[2])
    want_yaw = math.degrees(math.atan2(-to[0], to[2]))
    want_pitch = -math.degrees(math.atan2(to[1], math.hypot(to[0], to[2])))
    dyaw = (want_yaw - yaw) % 360.
    if dyaw > 180.: dyaw -= 360.
    return want_pitch - pitch, dyaw


def aim_delta(eye, yaw, pitch, pos):
    """point_delta to the centre of block `pos`."""
    return point_delta(eye, yaw, pitch, (pos[0] + .5, pos[1] + .5, pos[2] + .5))


def _solid(geometry, pos):
    block = geometry.blocks.get(tuple(pos))
    return block is not None and block != 'air'


def first_solid(geometry, start, end):
    """The first solid fixture block on the straight segment from `start` to `end`, or None. Positions the fixture did
    not fill, and its air fill, do not block (the arena is a box the fixture builds, so that is all there is)."""
    cell = [math.floor(v) for v in start]
    last = tuple(math.floor(v) for v in end)
    delta = [e - s for s, e in zip(start, end)]
    step = [(d > 0) - (d < 0) for d in delta]
    t_next = [((c + (d > 0) - s) / d) if d else math.inf for s, d, c in zip(start, delta, cell)]
    t_step = [abs(1. / d) if d else math.inf for d in delta]
    while True:
        if _solid(geometry, cell): return tuple(cell)
        if tuple(cell) == last: return None
        axis = min(range(3), key=t_next.__getitem__)
        if t_next[axis] > 1.: return None
        cell[axis] += step[axis]; t_next[axis] += t_step[axis]


def sight_plan(s, view, target, geometry):
    """What the sight teacher does on a near row (line-of-sight design 2026-11-10 §3): ('centre', None) when a straight
    line from the eye reaches the target centre; ('face', point) when it only reaches an open face (the nearest such
    point, INSET inside the block); ('walk', (x, z)) when no face is in sight and there is room to stand STAND blocks in
    front of an open side face (the nearest such spot); else ('hidden', None)."""
    target = tuple(target)
    centre = tuple(c + .5 for c in target)
    if first_solid(geometry, view.eye, centre) == target: return 'centre', None
    faces = [n for n in NORMALS if not _solid(geometry, [c + k for c, k in zip(target, n)])]
    points = [tuple(c + (.5 - INSET) * k for c, k in zip(centre, n)) for n in faces]
    seen = [p for p in points if first_solid(geometry, view.eye, p) == target]
    if seen: return 'face', min(seen, key=lambda p: math.dist(p, view.eye))
    feet = math.floor(s.position[1] + .01)
    spots = []
    for n in faces:
        if n[1]: continue                                   # only side faces have a place to stand in front of them
        x, z = centre[0] + (.5 + STAND) * n[0], centre[2] + (.5 + STAND) * n[2]
        cell = (math.floor(x), feet, math.floor(z))
        if not _solid(geometry, cell) and not _solid(geometry, (cell[0], feet + 1, cell[2])): spots.append((x, z))
    if not spots: return 'hidden', None
    return 'walk', min(spots, key=lambda p: math.hypot(p[0] - s.position[0], p[1] - s.position[2]))


def in_view(dpitch, dyaw):
    """Rough on-screen test for diagnostics only: half angles of the 16:9 view (vertical 35, horizontal 50)."""
    return abs(dyaw) <= 50. and abs(dpitch) <= 35.


def horizontal(s, pos):
    """Blocks from the feet to the centre of block `pos`, ignoring height."""
    return math.hypot(pos[0] + .5 - s.position[0], pos[2] + .5 - s.position[2])


def target_of(s, view, wanted):
    """The block the label aims at: in collect the last broken block if it was a target (None otherwise), else the
    mined or aimed target block, else the nearest remaining one."""
    if view.stage == 'collect':
        ok = block_matches(s.last_destroyed, wanted) and s.last_destroyed_pos is not None
        return tuple(s.last_destroyed_pos) if ok else None
    if view.stage == 'mining' and block_matches(s.mining_block, wanted) and s.mining_pos is not None:
        return s.mining_pos
    if view.stage in ('aimed', 'mining') and block_matches(s.hit_block, wanted) and s.hit_pos is not None:
        return s.hit_pos
    return view.nearest


MINING_FEATURES = ('sin_yaw', 'cos_yaw', 'pitch', 'dist')
FEATURE_PITCH = 60.        # degrees of pitch per unit of the `pitch` feature
FEATURE_DIST = 9.          # blocks per unit of the `dist` feature (0..1)


def mining_features(s, view, wanted):
    """Privileged head input next to the stage (mining direction input design 2026-11-14 §1), in MINING_FEATURES order:
    the direction from the eye to the centre of the block the teacher aims at (target_of) as the sin and cos of the yaw
    difference and the pitch difference / 60, and the eye distance / 9. Zeros without a sample, a view or a target, or
    with a GUI open."""
    zeros = [0.] * len(MINING_FEATURES)
    if s is None or view is None or not s.valid or s.gui_open or view.stage == 'gui': return zeros
    target = target_of(s, view, wanted)
    if target is None: return zeros
    dpitch, dyaw = aim_delta(view.eye, s.yaw, s.pitch, target)
    dist = math.dist(view.eye, tuple(c + .5 for c in target))
    return [math.sin(math.radians(dyaw)), math.cos(math.radians(dyaw)), max(-1., min(1., dpitch / FEATURE_PITCH)),
            min(dist, FEATURE_DIST) / FEATURE_DIST]


AIM_MODES = ('centre', 'hold', 'approach', 'press', 'sight')
WALKING_MODES = ('approach', 'press', 'sight')
PRESS_MODES = ('press', 'sight')


def _turn(dpitch, dyaw):
    return np.array([max(-MAX_TURN, min(MAX_TURN, dpitch)), max(-MAX_TURN, min(MAX_TURN, dyaw))])


def teacher_env_action(s, view, wanted, aim='centre', geometry=None):
    """MineRL-style action label for one state, or None when the row gets no label.

    aim='centre' (design §4) turns toward the target block centre in every labelled stage. aim='hold' (condition
    added 2026-11-07 after rounds 1-2: the centring label became camera jitter that kept resetting block damage) keeps
    the view still while the crosshair is on the target family and only turns while near but not aimed.
    aim='approach' (design 2026-11-09 §3) is 'hold' plus forward while the target is farther than CLOSE, and labels
    collect rows: turn toward the broken target block and walk onto it, no attack.
    aim='press' (design 2026-11-10 §3) is 'approach' that holds attack and the view on every mining-stage row.
    aim='sight' (line-of-sight design 2026-11-10 §3) is 'press' whose near label follows sight_plan when `geometry`
    (the fixture blocks of the moment) is given; the other modes ignore `geometry`."""
    if aim not in AIM_MODES: raise ValueError('unknown teacher aim mode: %s' % aim)
    stages = APPROACH_STAGES if aim in WALKING_MODES else LABELLED_STAGES
    if not s.valid or s.gui_open or view.stage not in stages: return None
    target = target_of(s, view, wanted)
    if target is None: return None
    action = {k: 0 for k in BUTTONS}
    if view.stage == 'collect':
        dpitch, dyaw = aim_delta(view.eye, s.yaw, s.pitch, target)
        action['camera'] = _turn(dpitch, dyaw)
        action['forward'] = int(horizontal(s, target) > DROP_CLOSE and abs(dyaw) <= FRONT)
        return action
    on_target = block_matches(s.hit_block, wanted) or (aim in PRESS_MODES and view.stage == 'mining')
    plan, point = ('centre', None)
    if aim == 'sight' and geometry is not None and view.stage == 'near' and not on_target:
        plan, point = sight_plan(s, view, target, geometry)
    if plan == 'walk':                      # hidden target: turn the yaw toward the standing spot and walk there
        dpitch, _ = aim_delta(view.eye, s.yaw, s.pitch, target)
        _, dyaw = point_delta(view.eye, s.yaw, s.pitch, (point[0], view.eye[1], point[1]))
        action['camera'] = _turn(dpitch, dyaw)
        away = math.hypot(point[0] - s.position[0], point[1] - s.position[2]) > SPOT_CLOSE
        action['forward'] = int(away and abs(dyaw) <= FRONT)
        return action
    if aim != 'centre' and on_target:
        action['camera'] = np.array([0., 0.]); front = True
    elif plan == 'face':                    # the centre is hidden but an open face is in sight: aim at that face
        dpitch, dyaw = point_delta(view.eye, s.yaw, s.pitch, point)
        action['camera'] = _turn(dpitch, dyaw); front = abs(dyaw) <= FRONT
    else:
        dpitch, dyaw = aim_delta(view.eye, s.yaw, s.pitch, target)
        action['camera'] = _turn(dpitch, dyaw); front = abs(dyaw) <= FRONT
    action['attack'] = int(on_target)
    if aim in WALKING_MODES: action['forward'] = int(horizontal(s, target) > CLOSE and front)
    return action


def policy_label(env_action, mapper, transformer):
    """Official factored label (buttons, camera indices) of an env action, checked by a mapping round trip."""
    factored = transformer.env2policy({k: np.asarray(v) for k, v in env_action.items()})
    mapped = mapper.from_factored({k: v[None] for k, v in factored.items()})
    decoded = transformer.policy2env(mapper.to_factored(mapped))
    roundtrip = mapper.from_factored(transformer.env2policy(decoded))
    if any(not np.array_equal(mapped[k], roundtrip[k]) for k in mapped):
        raise ValueError('label failed official mapping roundtrip')
    camera = np.asarray(decoded['camera'], dtype=float).reshape(-1)[:2]
    return dict(buttons=int(np.asarray(mapped['buttons']).reshape(-1)[0]),
                camera=int(np.asarray(mapped['camera']).reshape(-1)[0]),
                camera_on=bool(np.any(camera != 0)), decoded_camera=camera)
