"""Auditable local navigation tasks; coordinates are privileged server observations."""
import math
from .waypoints import goal_features,arrived
from .rl_core import tick_delta


def distance(g,target):return math.hypot(g['server_x']-target[0],g['server_z']-target[2])


def motion_goal(current,target,previous=None,*,remaining=100):
    goal=goal_features(current,target)
    if type(remaining) is not int or not 0<=remaining<=100:raise ValueError('invalid remaining budget')
    dx=dz=turn=0.
    if previous is not None:
        tick_delta(previous,current);goal_features(previous,target)
        dx=current['server_x']-previous['server_x'];dz=current['server_z']-previous['server_z']
        turn=((current['yaw']-previous['yaw']+180)%360-180)/180
    yaw=math.radians(current['yaw'])
    lateral=dx*math.cos(yaw)+dz*math.sin(yaw)
    forward=-dx*math.sin(yaw)+dz*math.cos(yaw)
    return goal+[max(-1.,min(1.,lateral)),max(-1.,min(1.,forward)),turn,remaining/100.]


def navigation_reward(before,after,target,*,terminal,success):
    tick_delta(before,after);goal_features(before,target);goal_features(after,target)
    if success and (not terminal or not arrived(after,target)):raise ValueError('unverified success')
    first=-min(distance(before,target),16.)/4
    last=0. if terminal else -min(distance(after,target),16.)/4
    task=1. if success else (-1. if terminal else 0.)
    shaping=.2*(.99*last-first)
    return dict(task=task,shaping=shaping,time_cost=-.002,total=task+shaping-.002,
                potential_before=first,potential_after=last,terminal=bool(terminal),success=bool(success))


def validate_dataset(data,method):
    if (data.get('schema')!='navigation_trials_v1' or data.get('split')!='train' or
            not data.get('eligible') or not data.get('rows')):raise ValueError('ineligible navigation training data')
    if method not in ('bc','ppo'):raise ValueError('unknown navigation training method')
    if method=='ppo' and any(r.get('owner')!='student' for r in data['rows']):
        raise ValueError('teacher/neutral rows cannot train PPO')


def validate_bc_lineage(data,behavior,candidate):
    validate_dataset(data,'bc')
    for key in ('parent_sha256','embedding_sha256','model_sha256'):
        value=behavior.get(key)
        if (not isinstance(value,str) or len(value)!=64 or
                any(c not in '0123456789abcdef' for c in value) or value!=candidate.get(key)):
            raise ValueError('BC frozen feature lineage mismatch: '+key)


def navigation_returns(rows):
    from .rl_core import gae
    advantages=[];returns=[];segment=[];completed=set()
    for row in rows:
        if row['index'] in completed:raise ValueError('reused goal after terminal')
        if segment and row['index']!=segment[0]['index']:raise ValueError('goal changed without terminal')
        segment.append(row)
        part=row.get('navigation_reward')
        if not part:raise ValueError('missing navigation reward')
        if part['terminal']:
            a,r=gae([s['navigation_reward']['total'] for s in segment],[s['value'] for s in segment],
                0.,terminal=True,gamma=.99,lam=.95)
            advantages.extend(a);returns.extend(r);completed.add(row['index']);segment=[]
    if segment:raise ValueError('incomplete local goal is not a terminal failure')
    return advantages,returns


def correction_weights(goals):
    from collections import Counter
    if not goals:raise ValueError('empty correction labels')
    buckets=[]
    for g in goals:
        if len(g)!=10 or not all(math.isfinite(v) for v in g):raise ValueError('invalid correction goal')
        angle=abs(math.degrees(math.atan2(g[0],g[1])))
        buckets.append((0 if angle<=15 else 1 if angle<=45 else 2 if angle<=135 else 3,g[2]*16<=1))
    counts=Counter(buckets)
    weights=[min(3.,math.sqrt(len(goals)/counts[k])) for k in buckets]
    mean=sum(weights)/len(weights)
    return [w/mean for w in weights]


def local_targets(world,points):
    ox,oz=[v*16 for v in world['chunk']]
    if world.get('experimental_split') not in ('train','development') or world.get('kind')!='recovery':
        raise ValueError('reserved navigation fixture required')
    if not 1<=len(points)<=6:raise ValueError('bounded local task sequence required')
    result=[]
    for x,z in points:
        if not all(math.isfinite(v) for v in (x,z)) or not 1.5<=x<=5.5 or not 1.5<=z<=12.5:
            raise ValueError('goal outside known open training area')
        result.append([ox+x,201.,oz+(16-z if world['curriculum']['mirror_z'] else z)])
    return result


class TrialState:
    def __init__(self,targets,mode,*,student_ticks=32):
        if mode not in ('correction','evaluate','ppo'):raise ValueError('unknown trial mode')
        if type(student_ticks) is not int or not 1<=student_ticks<=32 or (mode!='correction' and student_ticks!=32):
            raise ValueError('student correction window only applies to explicit demonstrations')
        self.student_ticks=student_ticks
        self.targets,self.mode=targets,mode;self.index=0;self.owner='student';self.count=0
        self.receipts=[];self.sequence=0;self.begin=0;self.finished=False

    @property
    def target(self):return self.targets[self.index]

    def start(self,g):
        while self.index<len(self.targets) and arrived(g,self.target):
            self.receipts.append(dict(index=self.index,outcome='already_satisfied',actions=0,student_success=False))
            self.index+=1
        self.finished=self.index==len(self.targets)

    def advance(self,before,after):
        if self.finished:raise RuntimeError('action after schedule completion')
        tick_delta(before,after);self.sequence+=1
        if self.owner=='neutral':
            if after.get('held_key_count')!=0 or after.get('screen')!=0:raise ValueError('teacher handoff input release unverified')
            self.owner='teacher';self.count=0;return None
        self.count+=1;success=arrived(after,self.target);terminal=success or self.count>=100
        result=navigation_reward(before,after,self.target,terminal=terminal,success=success)
        if terminal:
            self.receipts.append(dict(index=self.index,target=self.target,outcome='completed' if success else 'failed',
                owner=self.owner,actions=self.sequence-self.begin,student_success=success and self.owner=='student',
                sequence=self.sequence,final_distance=distance(after,self.target)))
            self.index+=1;self.owner='student';self.count=0;self.begin=self.sequence;self.start(after)
        elif self.mode=='correction' and self.owner=='student' and self.count>=self.student_ticks:self.owner='neutral'
        return result
