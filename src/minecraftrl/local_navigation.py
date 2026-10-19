"""Conservative walking routes from bounded live server observations, never saved NBT.

The planner selects coordinates only. The existing learned actor owns every motor action.
Unknown, wet, unsupported and non-air body cells are excluded, including diagonal corner cuts.
"""
from collections import Counter, deque
from copy import deepcopy
import math


SOURCE='privileged_local_passability_v1'


def checked_grid(obs):
    g=obs.get('local_navigation')
    if (not isinstance(g,dict) or g.get('source')!=SOURCE or g.get('radius')!=6
            or g.get('tick')!=obs.get('server_tick') or obs.get('server_available')!=1
            or obs.get('error')!=0 or obs.get('screen')!=0):
        raise ValueError('fresh local passability observation required')
    for a in 'xyz':
        if type(g.get(a)) not in (int,float) or not math.isfinite(g[a]) or int(g[a])!=g[a]:
            raise ValueError('invalid local grid origin')
    if (any(g[a]!=math.floor(obs['server_'+a]) for a in 'xz')
            or not -.01<=obs['server_y']-g['y']<=2.01):
        raise ValueError('local grid pose mismatch')
    for k in ('walkable','wood_access'):
        if not isinstance(g.get(k),list) or len(g[k])!=169 or any(type(v) not in (int,float) or v not in (0,1) for v in g[k]):
            raise ValueError('invalid local passability cells')
    if any(b and not a for a,b in zip(g['walkable'],g['wood_access'])):
        raise ValueError('wood access without walking support')
    return g


def cells(g):
    return [(int(g['x'])+dx,int(g['z'])+dz) for dz in range(-6,7) for dx in range(-6,7)]


def clear_segment(known,start,end):
    """Swept 0.6-block player footprint; conservative full air cells, same floor."""
    if abs(start[1]-end[1])>.2:return False
    n=max(1,math.ceil(math.hypot(end[0]-start[0],end[2]-start[2])/.1))
    for i in range(n+1):
        x=start[0]+(end[0]-start[0])*i/n;z=start[2]+(end[2]-start[2])*i/n
        if any(not known.get((math.floor(x+dx),math.floor(z+dz)),False)
               for dx in (-.299,.299) for dz in (-.299,.299)):
            return False
    return True


def validate_candidate(obs,target):
    g=checked_grid(obs);start=[obs['server_'+a] for a in 'xyz']
    start[1]=g['y']
    return (0.35<math.hypot(target[0]-start[0],target[2]-start[2])<=3.000001
            and target[1]==g['y']
            and clear_segment(dict(zip(cells(g),g['walkable'])),start,target))


class LocalNavigation:
    def __init__(self):
        self.floor=None;self.known={};self.access=set();self.target=None
        self.visits=Counter();self.last_cell=None;self.last_tick=None
        self.rejected={};self.latest=None

    def observe(self,obs):
        g=checked_grid(obs)
        if self.last_tick is not None and g['tick']<self.last_tick:raise ValueError('local map tick reversed')
        # A fresh supported ground plane remains useful while the player is airborne.
        if not g['walkable'][84]:return False
        if self.floor!=g['y']:
            self.floor=g['y'];self.known={};self.access=set();self.rejected={}
        target=tuple(obs['wood_v3_target_'+a] for a in 'xyz') if obs.get('wood_v3_kind')==1 else None
        if target!=self.target:self.target=target;self.access=set()
        for c,open_,access in zip(cells(g),g['walkable'],g['wood_access']):
            self.known[c]=bool(open_)
            self.access.discard(c)
            if access and target is not None:self.access.add(c)
        # Per-episode bounded memory; retain nearby observations if exploration grows large.
        if len(self.known)>4096:
            nearest=sorted(self.known,key=lambda c:abs(c[0]-g['x'])+abs(c[1]-g['z']))[:2048]
            self.known={c:self.known[c] for c in nearest};self.access.intersection_update(self.known)
        c=(int(g['x']),int(g['z']))
        if c!=self.last_cell:self.visits[c]+=1
        self.last_cell=c;self.last_tick=g['tick'];self.latest=deepcopy(g)
        return True

    def reject(self,target,tick):
        # A failed motor attempt is temporary evidence, not proof of impassable terrain.
        self.rejected[(math.floor(target[0]),math.floor(target[2]))]=tick+80

    def choices(self,obs):
        if not self.observe(obs):return []
        p=[float(obs['server_'+a]) for a in 'xyz'];start=(math.floor(p[0]),math.floor(p[2]))
        if self.floor is None or not -.01<=p[1]-self.floor<=2.01:return []
        p[1]=self.floor
        parent={start:None};depth={start:0};q=deque([start])
        while q:
            a=q.popleft()
            for dx,dz in ((0,-1),(1,0),(0,1),(-1,0)):
                b=(a[0]+dx,a[1]+dz)
                if b in parent or not self.known.get(b) or self.rejected.get(b,0)>obs['server_tick']:continue
                parent[b]=a;depth[b]=depth[a]+1;q.append(b)
        access=[c for c in self.access if c in parent]
        if access:
            goals=sorted(access,key=lambda c:(depth[c],self.visits[c],c));purpose='approach'
        else:
            goals=[c for c in parent if c!=start]
            def exploration_cost(c):
                unknown=sum((c[0]+dx,c[1]+dz) not in self.known for dx,dz in ((0,-1),(1,0),(0,1),(-1,0)))
                return 8*self.visits[c]-5*unknown-.3*min(depth[c],12)
            goals.sort(key=lambda c:(exploration_cost(c),depth[c],c));purpose='explore'
        choices=[];used=set();live=dict(zip(cells(self.latest),self.latest['walkable']))
        for end in goals:
            path=[end]
            while parent[path[-1]] is not None:path.append(parent[path[-1]])
            path.reverse();chosen=None
            for cell in path:
                target=[cell[0]+.5,self.floor,cell[1]+.5]
                dist=math.hypot(target[0]-p[0],target[2]-p[2])
                if dist>3 or not clear_segment(live,p,target):break
                if dist>.35:chosen=target
            if chosen is None or tuple(chosen) in used:continue
            used.add(tuple(chosen))
            choices.append(dict(target=chosen,cost=float(len(choices)),admissible=True,purpose=purpose,
                                destination=[end[0]+.5,self.floor,end[1]+.5],path=[list(c) for c in path]))
            if len(choices)==4:break
        return choices

    def waypoint(self,obs):
        choices=self.choices(obs)
        return choices[0] if choices else None
