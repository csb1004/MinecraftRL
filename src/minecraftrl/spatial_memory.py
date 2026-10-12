"""Observed visits and directed crossings. This is not an occupancy map or motor policy."""
from copy import deepcopy
import heapq
import json
import math
from pathlib import Path
import re

from .memory import atomic_json


SOURCES=('privileged_server','f3_ocr')
CELL_SIZE=(2.,4.,2.)


def xyz(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=3 or
            any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>30000000 for v in value)):
        raise ValueError('invalid coordinate observation')
    return tuple(float(v) for v in value)


class SpatialMemory:
    def __init__(self,world_id,episode_id,*,source,dimension):
        if source not in SOURCES or any(not isinstance(s,str) or not 1<=len(s)<=512 for s in (world_id,episode_id,dimension)):
            raise ValueError('explicit spatial observation scope required')
        self.world_id,self.episode_id,self.source,self.dimension=world_id,episode_id,source,dimension
        self.nodes={};self.edges={};self.landmarks={};self.last=None;self.current=None
        self.entries=0;self.discontinuities=0;self.samples=0;self.saturated=False

    def observe(self,position,tick):
        position=xyz(position)
        if type(tick) not in (int,float) or not math.isfinite(tick) or tick<0 or int(tick)!=tick:
            raise ValueError('invalid coordinate tick')
        if self.last is not None and tick<=self.last['tick']:raise ValueError('nonmonotonic coordinate tick')
        cell=tuple(math.floor(p/s) for p,s in zip(position,CELL_SIZE))
        if cell not in self.nodes and len(self.nodes)>=8192:
            self.saturated=True;self.last=None;self.current=None;return
        previous=self.current;contiguous=False
        if self.last is not None:
            dt=tick-self.last['tick']
            contiguous=dt<=4 and math.dist(position,self.last['position'])<=2*dt and all(abs(a-b)<=1 for a,b in zip(cell,previous))
            if not contiguous:self.discontinuities+=1
        node=self.nodes.setdefault(cell,dict(visits=0,last_tick=int(tick),position=list(position)))
        if previous!=cell or self.last is None:
            node['visits']+=1;self.entries+=1
        node.update(last_tick=int(tick),position=list(position))
        if contiguous and previous!=cell:
            edge=(previous,cell);self.edges[edge]=self.edges.get(edge,0)+1
        self.current=cell;self.last=dict(tick=int(tick),position=position);self.samples+=1

    def break_segment(self):
        self.last=None;self.current=None

    def set_landmark(self,name,cell=None):
        if not isinstance(name,str) or not re.fullmatch(r'[\w -]{1,48}',name):raise ValueError('invalid landmark name')
        cell=self.current if cell is None else self._cell(cell)
        if cell not in self.nodes:raise ValueError('landmark must be an observed location')
        if name not in self.landmarks and len(self.landmarks)>=32:raise ValueError('landmark capacity exceeded')
        self.landmarks[name]=list(cell)

    def remove_landmark(self,name):
        del self.landmarks[name]

    @staticmethod
    def _cell(value):
        if not isinstance(value,(list,tuple)) or len(value)!=3 or any(type(v) is not int for v in value):
            raise ValueError('invalid cell identity')
        return tuple(value)

    def route(self,target,*,mode='return',start=None):
        if mode not in ('explore','return','resource'):raise ValueError('invalid navigation purpose')
        start=self.current if start is None else self._cell(start);target=self._cell(target)
        if start not in self.nodes or target not in self.nodes:return None
        adjacency={}
        for (a,b),visits in self.edges.items():
            cost=math.sqrt(sum(((y-x)*size)**2 for x,y,size in zip(a,b,CELL_SIZE)))
            if mode=='explore':cost*=1+min(visits,8)
            adjacency.setdefault(a,[]).append((b,cost))
        queue=[(0.,start)];best={start:0.};parent={}
        while queue:
            cost,a=heapq.heappop(queue)
            if cost!=best[a]:continue
            if a==target:
                cells=[a]
                while cells[-1]!=start:cells.append(parent[cells[-1]])
                return dict(cells=[list(c) for c in reversed(cells)],cost=cost,mode=mode,
                    execution_authorized=False,scope='observed directed cell crossings; not a collision-free waypoint trajectory')
            for b,extra in adjacency.get(a,[]):
                candidate=cost+extra
                if candidate<best.get(b,float('inf')):
                    best[b]=candidate;parent[b]=a;heapq.heappush(queue,(candidate,b))
        return None

    def route_to_landmark(self,name,*,mode='return'):
        return self.route(self.landmarks[name],mode=mode)

    def summary(self):
        # Missing adjacent crossings indicate unknown connectivity, NOT open frontiers.
        candidates=[]
        if self.current is not None:
            nearby=sorted(self.nodes,key=lambda c:(self.nodes[c]['visits'],math.dist(c,self.current),c))[:24]
            for cell in nearby:
                unknown=[]
                for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                    other=(cell[0]+dx,cell[1],cell[2]+dz)
                    if (cell,other) not in self.edges:unknown.append([dx,dz])
                if unknown:
                    candidates.append(dict(cell=list(cell),visits=self.nodes[cell]['visits'],
                        unobserved_directions=unknown,traversability='unknown',executable=False))
        return dict(world_id=self.world_id,episode_id=self.episode_id,dimension=self.dimension,source=self.source,
            cell_size=list(CELL_SIZE),current=list(self.current) if self.current is not None else None,
            last_tick=self.last['tick'] if self.last is not None else None,
            unique_cells=len(self.nodes),directed_edges=len(self.edges),cell_entries=self.entries,
            samples=self.samples,discontinuities=self.discontinuities,saturated=self.saturated,
            landmarks=deepcopy(self.landmarks),inspection_candidates=candidates[:8],
            frontier_status='unverified_without_local_passability',waypoint_execution_supported=False,
            rule='Exploration prefers less repeated observed routes; return/resource allow familiar routes. Unknown directions are inspection hypotheses, never traversable paths.')

    def save(self,path):
        atomic_json(Path(path),dict(schema=1,world_id=self.world_id,episode_id=self.episode_id,
            source=self.source,dimension=self.dimension,cell_size=list(CELL_SIZE),entries=self.entries,
            samples=self.samples,discontinuities=self.discontinuities,saturated=self.saturated,
            nodes=[dict(cell=list(c),**v) for c,v in sorted(self.nodes.items())],
            edges=[dict(start=list(a),end=list(b),visits=v) for (a,b),v in sorted(self.edges.items())],
            landmarks=deepcopy(self.landmarks)))

    @classmethod
    def load(cls,path,*,world_id,source,dimension):
        path=Path(path)
        if path.stat().st_size>4*1024*1024:raise ValueError('spatial memory too large')
        data=json.loads(path.read_text(encoding='utf-8'))
        if (data.get('schema')!=1 or data.get('world_id')!=world_id or data.get('source')!=source or
                data.get('dimension')!=dimension or data.get('cell_size')!=list(CELL_SIZE)):
            raise ValueError('spatial memory scope mismatch')
        m=cls(world_id,data['episode_id'],source=source,dimension=dimension)
        if len(data['nodes'])>8192 or len(data['edges'])>8192*26:raise ValueError('spatial capacity exceeded')
        def count(v):
            if type(v) is not int or v<0:raise ValueError('invalid spatial counter')
            return v
        for item in data['nodes']:
            c=m._cell(item['cell']);position=xyz(item['position'])
            if c in m.nodes or tuple(math.floor(p/s) for p,s in zip(position,CELL_SIZE))!=c:raise ValueError('invalid saved cell')
            m.nodes[c]=dict(visits=count(item['visits']),last_tick=count(item['last_tick']),position=list(position))
        for item in data['edges']:
            a,b=m._cell(item['start']),m._cell(item['end']);edge=(a,b)
            if a not in m.nodes or b not in m.nodes or a==b or edge in m.edges or any(abs(x-y)>1 for x,y in zip(a,b)):
                raise ValueError('invalid saved crossing')
            m.edges[edge]=count(item['visits'])
            if not m.edges[edge]:raise ValueError('untraversed saved edge')
        for name,cell in data['landmarks'].items():m.set_landmark(name,cell)
        m.entries=count(data['entries']);m.samples=count(data['samples']);m.discontinuities=count(data['discontinuities'])
        if m.entries!=sum(v['visits'] for v in m.nodes.values()):raise ValueError('entry total mismatch')
        if type(data['saturated']) is not bool:raise ValueError('invalid saturation flag')
        m.saturated=data['saturated']
        # A restored episode starts a new observation segment, even if its tick resets.
        return m
