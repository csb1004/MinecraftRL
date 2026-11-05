"""Declared two-level training arenas and their privileged demonstration teacher.

This teacher is not a policy fallback or a natural-terrain path finder.
"""
import math

from .navigation_observation import position

SCHEMA = 'step_navigation_goals_v1'
MATERIALS = ('minecraft:stone', 'minecraft:dirt', 'minecraft:oak_leaves')


def terrain_action_count(rows):
    """Decode JSON button values; a one-element [0] is a released button."""
    def pressed(value):
        while isinstance(value, list):
            if len(value) != 1:
                raise ValueError('one scalar per terrain button required')
            value = value[0]
        if type(value) not in (int, float, bool) or value not in (0, 1):
            raise ValueError('binary terrain button required')
        return value == 1
    count = 0
    for row in rows:
        # Validate both buttons even when the first one is pressed.
        active = [pressed(row['action'][key]) for key in ('attack', 'use')]
        count += any(active)
    return count


def validate_layout(layout):
    required = {'material', 'rotation', 'edge', 'start_distance', 'yaw_offset'}
    if not isinstance(layout, dict) or set(layout) not in (required,required|{'goal_profile'},required|{'goal_profile','start_side'}):
        raise ValueError('explicit step layout fields required')
    if 'goal_profile' in layout and layout['goal_profile']!='stop-before-step':
        raise ValueError('unsupported step goal profile')
    if layout.get('start_side','lower') not in ('lower','upper'):
        raise ValueError('step start side must be lower or upper')
    if layout['material'] not in MATERIALS:
        raise ValueError('unsupported step material')
    if type(layout['rotation']) is not int or layout['rotation'] not in (0, 90, 180, 270):
        raise ValueError('step rotation must be a quarter turn')
    if type(layout['edge']) is not int or not 7 <= layout['edge'] <= 9:
        raise ValueError('step edge outside bounded curriculum')
    for key, low, high in (('start_distance', .35, 2.), ('yaw_offset', -45., 45.)):
        value = layout[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid step layout '+key)
    return layout


def rotate(x, z, rotation, *, block=False):
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('invalid rotation')
    end = 15 if block else 16
    for _ in range(rotation // 90):
        x, z = end-z, x
    return x, z


def geometry(layout):
    validate_layout(layout)
    edge, material, rotation = layout['edge'], layout['material'], layout['rotation']
    blocks = {(x, 200, z): material for x in range(16) for z in range(16)}
    for x in range(16):
        for z in range(16):
            if x in (0, 15) or z in (0, 15):
                for y in range(201, 205):
                    blocks[x, y, z] = 'minecraft:bedrock'
            elif x >= edge:
                blocks[x, 201, z] = material
    rotated = {}
    for (x, y, z), material in blocks.items():
        rx, rz = rotate(x, z, rotation, block=True)
        rotated[rx, y, rz] = material
    def point(x, y, z):
        rx, rz = rotate(x, z, rotation)
        return [float(rx), float(y), float(rz)]
    upper_start=layout.get('start_side','lower')=='upper'
    start = point(edge+(layout['start_distance'] if upper_start else -layout['start_distance']),
                  202 if upper_start else 201, 3.5)
    if layout.get('goal_profile')=='stop-before-step':
        points = [point(edge-.5, 201, 3.5), point(edge+.5, 202, 3.5),
                  point(edge-.5, 201, 6.5), point(edge-.5, 201, 10.5),
                  point(edge+.5, 202, 10.5), point(edge-.5, 201, 11.5)]
    else:
        points = [point(edge+.5, 202, 3.5), point(edge+1.5, 202, 6.5),
                  point(edge-1.5, 201, 6.5), point(edge+.5, 202, 10.5),
                  point(edge+1.5, 202, 11.5), point(edge-1.5, 201, 11.5)]
    for p in [start]+points:
        validate_clearance(rotated, p)
    yaw = ((90. if upper_start else -90.)+layout['yaw_offset']+rotation+180.) % 360.-180.
    return rotated, start, yaw, points


def validate_clearance(blocks, point):
    x, y, z = point
    if y not in (201, 202):
        raise ValueError('step feet must rest on a declared floor')
    for bx in range(math.floor(x-.3+1e-7), math.floor(x+.3-1e-7)+1):
        for bz in range(math.floor(z-.3+1e-7), math.floor(z+.3-1e-7)+1):
            if (bx, y-1, bz) not in blocks or any((bx, by, bz) in blocks for by in (y, y+1)):
                raise ValueError('step body intersects terrain or lacks support')


def step_contract(layout, chunk, archive_sha256, *, student_prefix_actions=8):
    if type(student_prefix_actions) is not int or not 1 <= student_prefix_actions <= 60:
        raise ValueError('step student prefix must be an integer from one to sixty')
    _, start, _, relative = geometry(layout)
    ox, oz = _origin(chunk)
    absolute = lambda p: [ox+p[0], p[1], oz+p[2]]
    schema=SCHEMA if student_prefix_actions==8 else 'step_navigation_goals_v2'
    return dict(schema=schema, coordinates='absolute_xyz', source_archive_sha256=archive_sha256,
                initial_position=absolute(start), points=[absolute(p) for p in relative],
                privileged_targets=True, privileged_teacher=True, terrain_modified=True,
                teacher_actions_per_goal=100, student_prefix_actions=student_prefix_actions, neutral_handoff_actions=1,
                maximum_correction_actions_per_goal=101+student_prefix_actions,
                evaluation_actions_per_goal=100, arrival_hold_ticks=12)


def _origin(chunk):
    if not isinstance(chunk, (list, tuple)) or len(chunk) != 2 or any(type(v) is not int for v in chunk):
        raise ValueError('integer source chunk required')
    return [v*16 for v in chunk]


def step_targets(world, points):
    if world.get('kind') != 'step_navigation' or world.get('experimental_split') not in ('train', 'development'):
        raise ValueError('explicit step training/development world required')
    sha = world.get('sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('step world hash required')
    contract=world.get('navigation_goals',{})
    if contract.get('schema') not in (SCHEMA,'step_navigation_goals_v2'):
        raise ValueError('unsupported step goal contract')
    prefix=8 if contract['schema']==SCHEMA else contract.get('student_prefix_actions')
    expected = step_contract(world.get('step_layout'), world.get('chunk'), sha,student_prefix_actions=prefix)
    if world.get('navigation_goals') != expected or world.get('position') != expected['initial_position']:
        raise ValueError('step geometry/world contract mismatch')
    if not isinstance(points, list) or points != expected['points']:
        raise ValueError('step goals differ from generated terrain')
    return expected['points']


def validate_step_trial(world, config):
    targets = step_targets(world, config.get('points'))
    if (config.get('goal_contract') != world['navigation_goals']['schema'] or config.get('pose_source') != 'client' or
            type(config.get('arrival_hold_ticks')) is not int or config['arrival_hold_ticks'] != 12 or
            config.get('mode') not in ('correction', 'evaluate') or
            config.get('split') != world['experimental_split'] or config.get('world_sha256') != world['sha256']):
        raise ValueError('step trial requires bound settled client-pose correction/evaluation')
    if config['mode'] == 'correction' and (config['split'] != 'train' or
            type(config.get('student_ticks')) is not int or
            config['student_ticks'] != world['navigation_goals']['student_prefix_actions']):
        raise ValueError('step demonstrations require training worlds and their bound student prefix')
    return targets


def step_teacher(gui, target, world, *, pose_source='client'):
    """Generate a bounded teacher request; actual arrival must still be observed."""
    from .fixture_probe import navigate
    targets = step_targets(world, world.get('navigation_goals', {}).get('points'))
    if pose_source != 'client' or list(target) not in targets:
        raise ValueError('step teacher requires declared target and current client pose')
    if gui.get('server_available') != 1 or gui.get('screen') != 0 or gui.get('error') != 0:
        raise ValueError('step teacher requires verified gameplay observation')
    x, y, z = position(gui, pose_source)
    ox, oz = _origin(world['chunk'])
    cx, cz = rotate(x-ox, z-oz, (360-world['step_layout']['rotation']) % 360)
    ground = gui.get('client_on_ground')
    if ground not in (0, 1) or type(ground) not in (int, float):
        raise ValueError('step teacher requires observed ground state')
    if not (1.3-1e-6 <= cx <= 14.7+1e-6 and 1.3-1e-6 <= cz <= 14.7+1e-6 and 200.95 <= y <= 203.5):
        raise ValueError('step teacher observation outside bounded arena')
    if any(type(gui.get(k)) not in (int, float) or not math.isfinite(gui[k]) for k in ('yaw', 'pitch')):
        raise ValueError('step teacher heading unavailable')
    # On the lip of a higher block, horizontal proximity alone can strand the
    # body above a lower target. Approach its supported center before releasing.
    radius=.05 if ground==1 and abs(y-target[1])>.05 else .22
    changes, arrived = navigate(gui, target[0], target[2], pose_source=pose_source,arrival_radius=radius)
    edge_distance = world['step_layout']['edge']-cx
    jump = (ground == 1 and abs(y-201.) <= .05 and target[1] == 202 and
            .25 <= edge_distance <= 1.2 and changes.get('forward') == 1)
    if changes:
        changes['jump'] = int(jump)
    return changes, arrived and abs(y-target[1]) <= .05
