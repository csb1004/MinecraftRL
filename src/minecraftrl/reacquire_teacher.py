"""Privileged turn-to-target teacher (design 2026-11-07 §4): labels only, never dispatched during play.

Uses the server pose and the fixture's target positions (privileged observations). For a policy row in the near,
aimed or mining stage the label turns the view toward the target block centre by at most MAX_TURN degrees per
axis, attacks only while the crosshair is on the target family, and presses nothing else.
"""
import math

import numpy as np

from .pipeline.steps import block_matches

LABELLED_STAGES = ('near', 'aimed', 'mining')
MAX_TURN = 10.
# steve1.VPT.lib.actions.Buttons.ALL (a test checks they match).
BUTTONS = ('attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint', 'use', 'drop', 'inventory',
           *('hotbar.%d' % i for i in range(1, 10)))


def aim_delta(eye, yaw, pitch, pos):
    """(dpitch, dyaw) in degrees from the current view to the centre of block `pos`.

    Same convention as stage_replay.view_angle: yaw 0 = +z, 90 = -x, pitch down positive. camera[0] adds to the
    pitch and camera[1] to the yaw. The yaw difference is wrapped to (-180, 180]."""
    to = (pos[0] + .5 - eye[0], pos[1] + .5 - eye[1], pos[2] + .5 - eye[2])
    want_yaw = math.degrees(math.atan2(-to[0], to[2]))
    want_pitch = -math.degrees(math.atan2(to[1], math.hypot(to[0], to[2])))
    dyaw = (want_yaw - yaw) % 360.
    if dyaw > 180.: dyaw -= 360.
    return want_pitch - pitch, dyaw


def in_view(dpitch, dyaw):
    """Rough on-screen test for diagnostics only: half angles of the 16:9 view (vertical 35, horizontal 50)."""
    return abs(dyaw) <= 50. and abs(dpitch) <= 35.


def target_of(s, view, wanted):
    """The block the label aims at: the mined or aimed target block, else the nearest remaining one."""
    if view.stage == 'mining' and block_matches(s.mining_block, wanted) and s.mining_pos is not None:
        return s.mining_pos
    if view.stage in ('aimed', 'mining') and block_matches(s.hit_block, wanted) and s.hit_pos is not None:
        return s.hit_pos
    return view.nearest


AIM_MODES = ('centre', 'hold')


def teacher_env_action(s, view, wanted, aim='centre'):
    """MineRL-style action label for one state, or None when the row gets no label.

    aim='centre' (design §4) turns toward the target block centre in every labelled stage. aim='hold' (condition
    added 2026-11-07 after rounds 1-2: the centring label became camera jitter that kept resetting block damage) keeps
    the view still while the crosshair is on the target family and only turns while near but not aimed."""
    if aim not in AIM_MODES: raise ValueError('unknown teacher aim mode: %s' % aim)
    if not s.valid or s.gui_open or view.stage not in LABELLED_STAGES: return None
    target = target_of(s, view, wanted)
    if target is None: return None
    action = {k: 0 for k in BUTTONS}
    on_target = block_matches(s.hit_block, wanted)
    if aim == 'hold' and on_target: action['camera'] = np.array([0., 0.])
    else:
        dpitch, dyaw = aim_delta(view.eye, s.yaw, s.pitch, target)
        action['camera'] = np.array([max(-MAX_TURN, min(MAX_TURN, dpitch)), max(-MAX_TURN, min(MAX_TURN, dyaw))])
    action['attack'] = int(on_target)
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
