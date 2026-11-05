"""Validate optional read-only hostile-target telemetry; unknown is never safe."""
import math


def active_threat(server):
    return active_scan(server,'threat',24)


def active_pursuit(server):
    visible=active_threat(server)
    pursuit=active_scan(server,'pursuit',32)
    if (server['pursuit_count']<server['threat_count'] or visible and
            server['pursuit_distance']>server['threat_distance']):
        raise ValueError('inconsistent_pursuit_observation')
    return pursuit


def active_scan(server,prefix,expected_radius):
    values=[server.get(prefix+'_'+k) for k in ('valid','scan_radius','count','distance')]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('invalid_'+prefix+'_observation')
    valid,radius,count,distance=values
    if (valid!=1 or radius!=expected_radius or count<0 or int(count)!=count
            or count==0 and distance!=-1 or count>0 and not 0<=distance<=radius):
        raise ValueError('invalid_'+prefix+'_observation')
    return count>0


def grounded(server):
    ground=server.get('server_on_ground');fall=server.get('server_fall_distance')
    if (type(ground) not in (int,float) or ground not in (0,1) or type(fall) not in (int,float)
            or not math.isfinite(fall) or fall<0):
        raise ValueError('invalid_ground_observation')
    return ground==1 and fall==0


class PursuitStability:
    """A bounded recovery verdict, never a motor controller or future safety guarantee."""
    def __init__(self):
        self.last_tick=None;self.last_health=None;self.quiet_since=None
        self.observed_pursuit_steps=0;self.quiet_ticks=0

    def observe(self,server):
        pursued=active_pursuit(server);grounded(server)
        tick=server.get('server_tick');health=server.get('server_health')
        if (type(tick) not in (int,float) or not math.isfinite(tick) or tick<0 or int(tick)!=tick
                or self.last_tick is not None and tick<=self.last_tick):
            raise ValueError('invalid recovery server tick')
        if type(health) not in (int,float) or not math.isfinite(health) or not 0<=health<=20:
            raise ValueError('invalid recovery health')
        self.observed_pursuit_steps+=int(pursued)
        gap=self.last_tick is not None and tick-self.last_tick!=1
        damage=self.last_health is not None and health<self.last_health-1e-5
        if pursued or damage or server.get('server_alive')!=1:
            self.quiet_since=None
        elif self.quiet_since is None or gap:self.quiet_since=tick
        self.quiet_ticks=0 if self.quiet_since is None else tick-self.quiet_since
        self.last_tick=tick;self.last_health=health

    def ready(self,server):
        # Cleanup can revoke an earlier verdict, but cannot accumulate unobserved quiet time.
        return (not active_pursuit(server) and grounded(server) and self.quiet_ticks>=100
                and server.get('server_alive')==1 and server.get('screen')==0
                and server.get('server_health',-1)>=self.last_health-1e-5)
