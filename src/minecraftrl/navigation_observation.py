"""Explicit control-position source; server completion evidence stays separate."""
import math
from .spatial_memory import xyz


def position(observation, source='server'):
    if source not in ('server', 'client'):
        raise ValueError('unknown navigation position source')
    if source == 'client':
        tick = observation.get('client_tick')
        if (observation.get('client_available') != 1 or not isinstance(tick, (int, float)) or
                not math.isfinite(tick) or tick < 0 or int(tick) != tick):
            raise ValueError('current client pose unavailable; no server fallback')
    return xyz([observation[source+'_'+axis] for axis in 'xyz'])


def client_tick_delta(before, after):
    position(before, 'client'); position(after, 'client')
    if after['client_tick'] - before['client_tick'] != 1:
        raise ValueError('navigation client observation must advance exactly one tick')
