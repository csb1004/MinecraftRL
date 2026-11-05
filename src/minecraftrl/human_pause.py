"""Observed movement-then-pause references; no intention or arrival supervision."""
import math

PAUSE_SCHEMA = 'observed_movement_pause_v1'
MOVEMENT = ('forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak')


def observed_pause_evidence(raw, converted, start, end):
    """Verify a 40-action clip ending in 12 released, position-stable transitions."""
    if (type(start) is not int or type(end) is not int or start < 0 or
            end - start != 40 or end >= len(raw)):
        raise ValueError('pause reference needs 40 actions and a final observed pose')
    part = raw[start:end + 1]
    for row in part:
        if row.get('isGuiOpen') is not False or type(row.get('tick')) is not int:
            raise ValueError('pause observations require closed GUI and integer ticks')
        if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
               for k in ('xpos', 'ypos', 'zpos')):
            raise ValueError('pause observations require finite recorded poses')
    if any(b['tick'] - a['tick'] != 1 for a, b in zip(part, part[1:])):
        raise ValueError('pause observations must be consecutive')
    for index in range(start, end):
        item = converted[index]
        if not item['bc_eligible']:
            raise ValueError('unsupported pause action')
        action = item['action']
        camera = action['camera']
        if (len(camera) != 2 or any(type(v) not in (int, float) or
                not math.isfinite(v) or abs(v) > 10 for v in camera)):
            raise ValueError('invalid or clipped pause camera')
        if any(action.get(key) not in (0, 1) for key in MOVEMENT):
            raise ValueError('invalid movement state')
        if any(value for key, value in action.items() if key not in (*MOVEMENT, 'camera')):
            raise ValueError('interaction is not movement-pause supervision')
    def horizontal(a, b):
        return math.hypot(a['xpos'] - b['xpos'], a['zpos'] - b['zpos'])
    distance = horizontal(part[0], part[-1])
    if not 1.5 <= distance <= 8:
        raise ValueError('pause needs a bounded observed approach')
    if any(math.dist([a[k] for k in ('xpos', 'ypos', 'zpos')],
                     [b[k] for k in ('xpos', 'ypos', 'zpos')]) > 1.25
           for a, b in zip(part, part[1:])):
        raise ValueError('discontinuous approach pose')
    pause = end - 12
    if not (any(horizontal(raw[i], raw[i - 1]) > .03 for i in range(pause - 8, pause)) and
            any(converted[i]['action'][key] for i in range(pause - 8, pause) for key in MOVEMENT[:4])):
        raise ValueError('pause needs observed approach motion immediately before release')
    if any(converted[i]['action'][key] for i in range(pause, end) for key in MOVEMENT):
        raise ValueError('pause movement buttons must be released')
    hsteps = [horizontal(raw[i + 1], raw[i]) for i in range(pause, end)]
    vsteps = [abs(raw[i + 1]['ypos'] - raw[i]['ypos']) for i in range(pause, end)]
    offsets = [horizontal(raw[i], raw[end]) for i in range(pause, end + 1)]
    heights = [abs(raw[i]['ypos'] - raw[end]['ypos']) for i in range(pause, end + 1)]
    if max(hsteps) > .03 or max(vsteps) > .01 or max(offsets) > .1 or max(heights) > .01:
        raise ValueError('recorded pause is not position-stable')
    return dict(schema=PAUSE_SCHEMA, pause_start=pause, end=end, observed_transitions=12,
        maximum_horizontal_step=max(hsteps), maximum_vertical_step=max(vsteps),
        maximum_horizontal_reference_error=max(offsets), maximum_vertical_reference_error=max(heights),
        movement_buttons_released=True, ground_observed=False, human_intent_observed=False,
        task_completion_observed=False, safety_observed=False)
