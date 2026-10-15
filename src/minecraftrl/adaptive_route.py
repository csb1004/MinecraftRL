"""Local failed-motion hypotheses, never an oracle occupancy map."""
from collections import deque
from copy import deepcopy
import math
from .exploration_route import ExplorationRoute,pose_only
from .waypoints import goal_features


class AdaptiveRoute(ExplorationRoute):
    def __init__(self,pose,history,*,blocked_directions=None,windowed_motion=False):
        from .motion_blockage import MotionBlockage
        self.motion_window=MotionBlockage() if windowed_motion else None;self.latest_blockage=None
        self.blocked_directions=deepcopy(blocked_directions or []);self.motion=deque(maxlen=12);self.interruptions=[];self.candidates=[]
        super().__init__(pose,history)

    @property
    def completed(self):return False  # Handoff is checked by the controller, not a goal counter.

    def blocked(self,g,target):
        p=self._position(g);dx,dz=target[0]-p[0],target[2]-p[2];n=math.hypot(dx,dz)
        return any(b['status']=='suspected' and math.hypot(p[0]-b['position'][0],p[2]-b['position'][2])<=2.5
            and ('window_actions' not in b or abs(p[1]-b['position'][1])<1.5)
            and n>0 and (dx*b['direction'][0]+dz*b['direction'][1])/n>=math.cos(math.radians(40))
            for b in self.blocked_directions)

    def observe_action(self,before,after,action,*,sequence):
        before=pose_only(before);after=pose_only(after);p=self._position(before);q=self._position(after)
        goal_features(before,p);goal_features(after,q);self._visit(q)
        for b in self.blocked_directions:
            dx,dz=q[0]-b['position'][0],q[2]-b['position'][2];ux,uz=b['direction']
            if (b['status']=='suspected' and math.hypot(p[0]-b['position'][0],p[2]-b['position'][2])<1.5
                    and ('window_actions' not in b or max(abs(p[1]-b['position'][1]),abs(q[1]-b['position'][1]))<1.5)
                    and dx*ux+dz*uz>=.65 and abs(dx*uz-dz*ux)<.6):
                b.update(status='refuted_by_crossing',refuted_sequence=sequence)
        if self.motion_window is not None:
            self.latest_blockage=self.motion_window.observe(before,after,action,sequence=sequence)
            if self.latest_blockage:
                d=self.latest_blockage['direction'];target=[q[0]+3*d[0],q[1],q[2]+3*d[1]]
                if not self.blocked(after,target) and len(self.blocked_directions)<32:
                    self.blocked_directions.append(deepcopy(self.latest_blockage))
            return
        scalar=lambda key:float(action.get(key,[0])[0] if isinstance(action.get(key,[0]),list) else action.get(key,0))
        f=scalar('forward')-scalar('back');l=scalar('left')-scalar('right');yaw=math.radians(before['yaw'])
        dx,dz=-math.sin(yaw)*f+math.cos(yaw)*l,math.cos(yaw)*f+math.sin(yaw)*l;n=math.hypot(dx,dz)
        if not n or abs(q[1]-p[1])>.25 or math.hypot(q[0]-p[0],q[2]-p[2])>.025:
            self.motion.clear();return
        direction=[dx/n,dz/n]
        if self.motion:
            first=self.motion[0]
            if (math.dist([p[0],p[2]],first['xz'])>.2 or sum(a*b for a,b in zip(direction,first['direction']))<.94):self.motion.clear()
        self.motion.append(dict(xz=[p[0],p[2]],direction=direction))
        if len(self.motion)==12:
            target=[p[0]+3*direction[0],p[1],p[2]+3*direction[1]]
            if not self.blocked(before,target) and len(self.blocked_directions)<32:
                self.blocked_directions.append(dict(position=q,direction=direction,support_actions=12,sequence=sequence,
                    status='suspected',scope='aligned movement commands with low observed displacement; not certified terrain'))
            self.motion.clear()

    def _new_goal(self,g,sequence):
        g=pose_only(g);position=self._position(g);goal_features(g,position);self._visit(position)
        if len(self.targets)>=8:self.exhausted=True;return
        basis=g['yaw'] if self.heading is None else self.heading
        offsets=(90.,-90.,180.,0.) if self.heading is None else (0.,90.,-90.,180.)
        candidates=[]
        for order,offset in enumerate(offsets):
            angle=basis+offset;radians=math.radians(angle)
            target=[position[0]-3*math.sin(radians),position[1],position[2]+3*math.cos(radians)]
            cell=self._cell(target);blocked=self.blocked(g,target)
            candidates.append(dict(target=target,heading=angle,visits=self.visits[cell],failed=self.failed[cell],
                cost=4*self.visits[cell]+12*self.failed[cell]+order*.01,admissible=not blocked))
        legal=[i for i,c in enumerate(candidates) if c['admissible']]
        self.candidates=deepcopy(candidates)
        if not legal:self.exhausted=True;return
        chosen=min(legal,key=lambda i:candidates[i]['cost']);best=candidates[chosen]
        self.index=len(self.targets);self.targets.append(best['target']);self.heading=best['heading']
        self.attempt_start=self.progress_sequence=sequence;self.best_distance=3.
        self.decisions.append(dict(sequence=sequence,pose=g,candidates=candidates,selected=chosen,
            source='adaptive_rule',traversability='unknown',blocked_records=len(self.blocked_directions)))

    def choose(self,index,*,sequence,source):
        if (self.exhausted or type(index) is not int or sequence!=self.attempt_start
                or not 0<=index<len(self.decisions[-1]['candidates']) or
                not self.decisions[-1]['candidates'][index]['admissible']):raise ValueError('stale or inadmissible route choice')
        best=self.decisions[-1]['candidates'][index]
        self.targets[self.index]=list(best['target']);self.heading=best['heading']
        self.decisions[-1].update(selected=index,source=source)

    def replan(self,g,*,sequence,reason):
        # A controller handoff ends an attempt without proving its target unreachable.
        # observe() alone records navigation timeout/no-progress failures.
        self.interruptions.append(dict(index=self.index,target=self.targets[self.index],sequence=sequence,
            actions=sequence-self.attempt_start,best_progress_distance=self.best_distance,reason=reason))
        self._new_goal(g,sequence)
