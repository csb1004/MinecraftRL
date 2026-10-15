"""Rank speculative local goals from observed visits; never infer free space."""
from copy import deepcopy
import math

from .spatial_memory import CELL_SIZE,xyz


def _cell(position):
    return tuple(math.floor(p/s) for p,s in zip(position,CELL_SIZE))


def score_candidates(memory,candidates,*,purpose='explore'):
    if purpose not in ('explore','return','resource'):raise ValueError('invalid exploration purpose')
    if memory.last is None or memory.current is None or memory.saturated:
        raise ValueError('current continuous spatial observation required')
    start=xyz(memory.last['position']);ranked=deepcopy(candidates);evidence=[]
    for index,c in enumerate(ranked):
        target=xyz(c['target']);length=math.dist(start,target)
        if not 0<length<=8 or type(c['admissible']) is not bool:
            raise ValueError('invalid local exploration candidate')
        failures=c.get('failed',0)
        if type(failures) is not int or failures<0:raise ValueError('invalid observed failure count')
        # Sampling is only a planning hypothesis. No node/edge is added to memory.
        cells=[];samples=max(1,math.ceil(length/.25))
        for i in range(samples+1):
            cell=_cell([a+(b-a)*i/samples for a,b in zip(start,target)])
            if not cells or cell!=cells[-1]:cells.append(cell)
        new=sum(cell not in memory.nodes for cell in set(cells[1:]))
        repeat=sum(memory.edges.get((a,b),0) for a,b in zip(cells,cells[1:]))
        visits=memory.nodes.get(cells[-1],{}).get('visits',0)
        score=12*failures+index*.01
        if purpose=='explore':score+=4*visits+2*repeat-3*new
        c['cost']=score
        evidence.append(dict(target=list(target),sampled_cells=[list(cell) for cell in cells],
            estimated_unvisited_cells=new,repeated_crossings=repeat,target_visits=visits,
            observed_failures=failures,cost=score,admissible=c['admissible']))
    summary=memory.summary()
    return ranked,dict(schema=1,purpose=purpose,observation={k:summary[k] for k in
        ('world_id','episode_id','dimension','source','last_tick','unique_cells','directed_edges')},
        traversability='unknown',sample_spacing=.25,candidates=evidence,
        rule='explore: 4*target_visits + 2*repeated_crossings - 3*unvisited_cells + 12*failures + index*.01; return/resource: no novelty preference')


def coverage_delta(before,after):
    for key in ('world_id','episode_id','dimension','source','cell_size'):
        if before[key]!=after[key]:raise ValueError('coverage observation scope changed')
    counts={}
    for name,key in (('new_cells','unique_cells'),('new_directed_crossings','directed_edges'),('cell_entries','cell_entries')):
        if any(type(v) is not int or v<0 for v in (before[key],after[key])) or after[key]<before[key]:
            raise ValueError('nonmonotonic coverage count')
        counts[name]=after[key]-before[key]
    return dict(**counts,source=after['source'],root_success_claimed=False,
        scope='Observed position coverage only; distinct from arrival, target discovery and inventory gain')
