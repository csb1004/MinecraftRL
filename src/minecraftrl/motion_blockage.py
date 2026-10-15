"""Short-window movement evidence; records a hypothesis, never a terrain map."""
from collections import deque
import math


class MotionBlockage:
    def __init__(self):self.samples=deque(maxlen=24);self.last=None

    def observe(self,before,after,action,*,sequence):
        if type(sequence) is not int or sequence<0 or (self.last is not None and sequence<=self.last):
            raise ValueError('nonmonotonic motion evidence')
        if self.last is not None and sequence!=self.last+1:self.samples.clear()
        self.last=sequence
        p=[float(before['server_'+a]) for a in 'xyz'];q=[float(after['server_'+a]) for a in 'xyz']
        if not all(math.isfinite(v) for v in p+q+[before['yaw']]):raise ValueError('invalid motion observation')
        scalar=lambda key:float(action.get(key,[0])[0] if isinstance(action.get(key,[0]),list) else action.get(key,0))
        f=scalar('forward')-scalar('back');side=scalar('left')-scalar('right');yaw=math.radians(before['yaw'])
        dx,dz=-math.sin(yaw)*f+math.cos(yaw)*side,math.cos(yaw)*f+math.sin(yaw)*side
        length=math.hypot(dx,dz);direction=[dx/length,dz/length] if length else None
        self.samples.append(dict(position=p,end=q,direction=direction,sequence=sequence,
            low_motion=math.hypot(q[0]-p[0],q[2]-p[2])<=.025))
        if direction is None or not self.samples[-1]['low_motion']:return None
        support=[s for s in self.samples if s['direction'] is not None and s['low_motion']
            and sum(a*b for a,b in zip(direction,s['direction']))>=.94
            and math.hypot(s['position'][0]-q[0],s['position'][2]-q[2])<=.2
            and abs(s['position'][1]-q[1])<1.5 and abs(s['end'][1]-q[1])<1.5]
        if len(support)<12:return None
        return dict(position=q,direction=direction,support_actions=len(support),sequence=sequence,
            support_sequences=[s['sequence'] for s in support],window_actions=len(self.samples),
            status='suspected',terrain_proven=False,
            scope='At least 12 similarly directed low-displacement commands within 24 committed actions near one position')


def target_blockage(rows,target):
    detector=MotionBlockage();record=None
    for index,row in enumerate(rows,1):
        record=detector.observe(row['server_before'],row['server_after'],row['action'],sequence=index)
    if record is None:return None
    p=record['position'];dx,dz=target[0]-p[0],target[2]-p[2];length=math.hypot(dx,dz)
    if length and (dx*record['direction'][0]+dz*record['direction'][1])/length>=math.cos(math.radians(40)):
        return record
    return None
