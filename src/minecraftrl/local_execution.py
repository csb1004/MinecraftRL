"""Opt-in local navigation timing and neural decoding; no motor action overrides."""

LOCAL_STALL_WINDOWS=dict(search=80,approach=64,mine=240,collect=80,complete=80)
LOCAL_GOAL_WAIT=160
MINING_TEMPERATURE=.35


def configure_stall(detector,stage):
    detector.windows=dict(LOCAL_STALL_WINDOWS)
    detector.goal_budget_ticks=LOCAL_GOAL_WAIT
    detector.window=detector.windows[stage]


def mining_temperature(obs,phase):
    # A remembered target alone is insufficient. Use current server mining plus
    # the current crosshair block, so looking away immediately restores sampling.
    if (phase not in ('collect','resumed') or obs.get('server_alive')!=1
            or obs.get('server_available')!=1 or obs.get('screen')!=0 or obs.get('error')!=0
            or obs.get('wood_v3_kind')!=1 or obs.get('wood_server_mining')!=1
            or obs.get('wood_v3_reachable')!=1):return 1.
    target=[obs.get('wood_v3_target_'+a) for a in 'xyz']
    if (None in target or target!=[obs.get('wood_mining_'+a) for a in 'xyz']
            or target!=[obs.get('hit_'+a) for a in 'xyz']):return 1.
    return MINING_TEMPERATURE
