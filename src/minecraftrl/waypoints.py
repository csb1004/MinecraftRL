"""Relative navigation goals and labels from explicit, audited teacher waypoints."""
import math
from .spatial_memory import xyz

GOAL_DIM=6
ARRIVAL_RADIUS=.35


def goal_features(observation,target):
    target=xyz(target);position=xyz([observation['server_'+v] for v in 'xyz'])
    yaw,pitch=float(observation['yaw']),float(observation['pitch'])
    if not math.isfinite(yaw) or not math.isfinite(pitch) or abs(pitch)>90:
        raise ValueError('invalid navigation orientation')
    if observation.get('server_available')!=1 or observation.get('error')!=0:
        raise ValueError('navigation observation unavailable')
    dx,dy,dz=[b-a for a,b in zip(position,target)]
    distance=math.hypot(dx,dz);angle=math.atan2(-dx,dz)-math.radians(yaw)
    return [math.sin(angle),math.cos(angle),min(distance/16,1.),max(-1.,min(1.,dy/4)),pitch/90,1.]


def arrived(observation,target):
    goal_features(observation,target)
    return (observation.get('screen')==0 and
        math.hypot(target[0]-observation['server_x'],target[2]-observation['server_z'])<=ARRIVAL_RADIUS and
        abs(target[1]-observation['server_y'])<=1.5)


def world_targets(world,waypoints):
    ox,oz=[v*16 for v in world['chunk']]
    return [[float(ox+x),201.,float(oz+(16-z if world['curriculum']['mirror_z'] else z))] for x,z in waypoints]


def label_waypoints(session,world,rows):
    handoff=session['handoffs'][0];targets=world_targets(world,handoff['waypoints'])
    current=0;labels={}
    for i,row in enumerate(rows):
        if i<handoff['step']:
            if row.get('loss_eligible'):raise ValueError('student prefix is not expert supervision')
            continue
        if row.get('owner')!='teacher' or not row.get('loss_eligible'):
            raise ValueError('teacher boundary mismatch')
        if current==len(targets):continue
        if row.get('phase')!='move':raise ValueError('teacher left navigation before reaching its route')
        target=targets[current];g=row['server_before']
        distance=math.hypot(target[0]-g['server_x'],target[2]-g['server_z'])
        if distance<.22:
            current+=1  # Original teacher spends this one transition releasing input.
        elif distance>ARRIVAL_RADIUS:
            labels[i]=dict(target=target,goal=goal_features(g,target),waypoint_index=current)
    if current!=len(targets):raise ValueError('incomplete demonstrated waypoint route')
    return labels


class RouteProgress:
    def __init__(self,targets):
        if not 1<=len(targets)<=8:raise ValueError('bounded explicit waypoint route required')
        self.targets=[list(xyz(t)) for t in targets];self.index=0;self.receipts=[];self.last_sequence=0

    @property
    def completed(self):return self.index==len(self.targets)

    def observe(self,g,*,sequence,stopped=False):
        if stopped:return None
        while not self.completed and arrived(g,self.targets[self.index]):
            self.receipts.append(dict(index=self.index,target=self.targets[self.index],sequence=sequence,
                actions=sequence-self.last_sequence,server_position=[g['server_'+a] for a in 'xyz']))
            self.last_sequence=sequence;self.index+=1
        return None if self.completed else self.targets[self.index]
