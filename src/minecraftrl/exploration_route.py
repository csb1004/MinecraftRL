"""Speculative position goals from visited cells; no world geometry or resource coordinates."""
from collections import Counter
import math
from .waypoints import arrived, goal_features

POSE_KEYS=('server_x','server_y','server_z','yaw','pitch','screen','error','server_available')


def pose_only(observation):
    return {key:observation[key] for key in POSE_KEYS}


class ExplorationRoute:
    def __init__(self,pose,history):
        self.targets=[];self.index=0;self.receipts=[];self.failures=[];self.decisions=[]
        self.visits=Counter();self.failed=Counter();self.last_cell=None
        self.last_sequence=0;self.attempt_start=0;self.progress_sequence=0
        self.exhausted=False;self.heading=None
        for position in history:self._visit(position)
        self._new_goal(pose_only(pose),0)

    @staticmethod
    def _position(g):return [float(g['server_'+key]) for key in 'xyz']

    @staticmethod
    def _cell(p):return (math.floor(p[0]/2),math.floor(p[2]/2))

    def _visit(self,position):
        if not all(math.isfinite(v) for v in position):raise ValueError('nonfinite visit')
        cell=self._cell(position)
        if cell!=self.last_cell:self.visits[cell]+=1
        self.last_cell=cell

    @property
    def completed(self):return len(self.receipts)>=3

    def _new_goal(self,g,sequence):
        position=self._position(g);goal_features(g,position);self._visit(position)
        if len(self.targets)>=8:self.exhausted=True;return
        basis=g['yaw'] if self.heading is None else self.heading
        offsets=(90.,-90.,180.,0.) if self.heading is None else (0.,90.,-90.,180.)
        candidates=[]
        for order,offset in enumerate(offsets):
            angle=basis+offset;radians=math.radians(angle)
            target=[position[0]-3*math.sin(radians),position[1],position[2]+3*math.cos(radians)]
            cell=self._cell(target)
            cost=4*self.visits[cell]+12*self.failed[cell]+order*.01
            candidates.append(dict(target=target,heading=angle,visits=self.visits[cell],failed=self.failed[cell],cost=cost))
        chosen=min(range(len(candidates)),key=lambda i:candidates[i]['cost']);best=candidates[chosen]
        self.index=len(self.targets);self.targets.append(best['target']);self.heading=best['heading']
        self.attempt_start=self.progress_sequence=sequence;self.best_distance=3.
        self.decisions.append(dict(sequence=sequence,pose=g,candidates=candidates,selected=chosen,
            source='own pose and observed visit/failure history only',traversability='unknown'))

    def observe(self,observation,*,sequence,stopped=False):
        if stopped:return None
        if type(sequence) is not int or sequence<self.last_sequence:raise ValueError('invalid exploration sequence')
        self.last_sequence=sequence
        if self.completed or self.exhausted:return None
        g=pose_only(observation);position=self._position(g);target=self.targets[self.index]
        goal_features(g,target);self._visit(position)
        distance=math.hypot(target[0]-position[0],target[2]-position[2])
        if arrived(g,target):
            self.receipts.append(dict(index=self.index,target=target,sequence=sequence,
                actions=sequence-self.attempt_start,server_position=position))
            if not self.completed:self._new_goal(g,sequence)
        else:
            if distance<=self.best_distance-.25:self.best_distance=distance;self.progress_sequence=sequence
            expired=sequence-self.attempt_start>=100
            stuck=sequence-self.progress_sequence>=40
            if expired or stuck:
                self.failures.append(dict(index=self.index,target=target,sequence=sequence,
                    actions=sequence-self.attempt_start,best_progress_distance=self.best_distance,
                    reason='candidate_budget' if expired else 'candidate_no_progress'))
                self.failed[self._cell(target)]+=1;self._new_goal(g,sequence)
        return None if self.completed or self.exhausted else self.targets[self.index]
