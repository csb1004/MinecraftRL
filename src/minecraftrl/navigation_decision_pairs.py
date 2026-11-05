"""Training-only goal contrasts on qualified procedural step observations."""
import random
from collections import defaultdict
import torch
from .navigation_goal_queries import supervision_goal

DECISIONS=('jump','forward')
OBJECTIVES=('bce','conditional-ranking')


def annotate_views(rows,labels,run,*,split):
    if split!='train' or not isinstance(run,str) or not run or len(rows)!=len(labels):
        raise ValueError('matching train-only view provenance required')
    for row,label in zip(rows,labels):
        if row.get('bc_source')!='procedural_step' or type(label.get('row')) is not int or label['row']<0:
            raise ValueError('qualified procedural step rows required')
    for row,label in zip(rows,labels):
        row.update(bc_step_view=(run,label['row']),bc_step_split=split)


def _mapping(physical_buttons):
    bits=torch.as_tensor(physical_buttons)
    if (bits.ndim!=2 or bits.shape[1]!=2 or bits.shape[0]<2 or
            not ((bits==0)|(bits==1)).all()):
        raise ValueError('binary jump/forward mapping required')
    bits=bits.bool()
    if not (bits.any(0)&(~bits).any(0)).all():raise ValueError('both classes required per decision')
    return bits


def decision_views(rows,physical_buttons):
    bits=_mapping(physical_buttons).cpu();groups=defaultdict(list)
    for index,row in enumerate(rows):
        if row.get('bc_source')!='procedural_step':continue
        view=row.get('bc_step_view')
        if (row.get('bc_step_split')!='train' or not isinstance(view,tuple) or len(view)!=2 or
                not isinstance(view[0],str) or type(view[1]) is not int or view[1]<0):
            raise ValueError('train-only qualified view identity required')
        action=row.get('bc_action',{}).get('buttons')
        if (not isinstance(action,torch.Tensor) or action.dtype!=torch.long or action.numel()!=1 or
                not 0<=int(action.item())<bits.shape[0]):
            raise ValueError('one valid official action label required')
        supervision_goal(row);groups[view].append(index)
    pools={name:[] for name in DECISIONS}
    for view,indices in sorted(groups.items()):
        if len(indices)!=6 or sum(not rows[i].get('bc_step_goal_query',False) for i in indices)!=1:
            raise ValueError('one observed and five alternative goals per view required')
        if any(rows[i]['features'] is not rows[indices[0]]['features'] for i in indices):
            raise ValueError('paired goals must share the identical frozen observation')
        goals=[tuple(supervision_goal(rows[i])[:6]) for i in indices]
        if len(set(goals))!=6:raise ValueError('six distinct spatial goals per view required')
        for column,name in enumerate(DECISIONS):
            order=lambda i:(supervision_goal(rows[i])[2],i)
            positive=sorted((i for i in indices if bits[int(rows[i]['bc_action']['buttons'].item()),column]),key=order)
            negative=sorted((i for i in indices if not bits[int(rows[i]['bc_action']['buttons'].item()),column]),key=order)
            if positive and negative:pools[name].append(dict(view=view,positive=positive,negative=negative))
    if not all(pools.values()):raise ValueError('both decision contrast populations required')
    return pools,dict(views={name:len(pool) for name,pool in pools.items()},
        qualified_observations=len(groups),seed=17603,rows_per_update=4,budget=1.,
        selection='Persistent shuffled view queues; nearest goals first, then cyclic positive/negative combinations',
        scope='Train-only additional unexecuted goal queries; physical-button decision pairs')


class DecisionPairSampler:
    def __init__(self,pools,seed=17603):
        if set(pools)!=set(DECISIONS) or not all(pools.values()) or type(seed) is not int:
            raise ValueError('two nonempty decision pools and integer seed required')
        self.pools=pools;self.rng=random.Random(seed);self.draws=0
        self.queues={name:[] for name in DECISIONS};self.positions={name:0 for name in DECISIONS}
        self.visits={name:[0]*len(pools[name]) for name in DECISIONS}

    def draw(self):
        result=[]
        for name in DECISIONS:
            if self.positions[name]==len(self.queues[name]):
                self.queues[name]=list(range(len(self.pools[name])));self.rng.shuffle(self.queues[name]);self.positions[name]=0
            selected=self.queues[name][self.positions[name]];self.positions[name]+=1
            pool=self.pools[name][selected];n=self.visits[name][selected]
            positive,negative=pool['positive'],pool['negative']
            result.extend([positive[n%len(positive)],negative[(n//len(positive))%len(negative)]])
            self.visits[name][selected]+=1
        self.draws+=1
        return result

    def report(self):
        return dict(draws=self.draws,views={name:dict(total=len(visits),seen=sum(v>0 for v in visits),
            minimum_visits=min(visits),maximum_visits=max(visits)) for name,visits in self.visits.items()})


def pair_rows(rows,indices):
    if len(indices)!=4 or any(type(i) is not int or not 0<=i<len(rows) for i in indices):
        raise ValueError('two valid positive/negative index pairs required')
    result=[]
    for index in indices:
        row=rows[index];goal=list(supervision_goal(row));goal[-1]=1.
        result.append(dict(row,bc_motion_goal=goal))
    return result


class DecisionPairBC(torch.nn.Module):
    def __init__(self,physical_buttons,objective='bce'):
        super().__init__();self.register_buffer('bits',_mapping(physical_buttons))
        if objective not in OBJECTIVES:raise ValueError('unsupported decision-pair objective')
        self.objective=objective

    def forward(self,button_logits):
        if (button_logits.ndim<2 or button_logits.shape[-1]!=self.bits.shape[0] or
                button_logits.numel()!=4*self.bits.shape[0] or not torch.isfinite(button_logits).all()):
            raise ValueError('finite joint logits for two decision pairs required')
        logits=button_logits.reshape(4,-1)
        if self.objective=='conditional-ranking':
            terms=[]
            for column in range(2):
                mask=self.bits[:,column];pair=logits[2*column:2*column+2]
                logodds=torch.logsumexp(pair[:,mask],-1)-torch.logsumexp(pair[:,~mask],-1)
                # Two queried inputs per pair; neutral-point gradients match four-row BCE.
                terms.append(.5*torch.nn.functional.softplus(logodds[1]-logodds[0]))
            return torch.stack(terms).mean()
        logp=logits-torch.logsumexp(logits,-1,keepdim=True)
        terms=[]
        for column in range(2):
            mask=self.bits[:,column]
            terms.extend([-torch.logsumexp(logp[2*column,mask],-1),
                          -torch.logsumexp(logp[2*column+1,~mask],-1)])
        return torch.stack(terms).mean()
