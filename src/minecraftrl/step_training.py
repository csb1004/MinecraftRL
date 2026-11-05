"""Explicit flat/step/human BC mixture with preserved old-population retention."""
import math
from collections import Counter

GROUPS=('flat','step','human')
SHARES={'flat':.5,'step':.25,'human':.25}
BATCH_COUNTS={'flat':16,'step':8,'human':8}


def supervision_groups(rows):
    mapping={None:'flat','procedural_step':'step','hindsight_reference':'human'}
    groups=[]
    for row in rows:
        if row.get('bc_source') not in mapping:raise ValueError('unknown supervision source')
        groups.append(mapping[row.get('bc_source')])
    _counts(groups)
    return groups


def _counts(groups,*,require_all=True):
    if not groups or any(g not in GROUPS for g in groups):raise ValueError('invalid supervision groups')
    counts=Counter(groups)
    if require_all and set(counts)!=set(GROUPS):raise ValueError('flat, step and human supervision are all required')
    return counts


def balanced_weights(weights,groups):
    _counts(groups)
    if len(weights)!=len(groups) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('positive finite weight per supervision row required')
    totals={g:sum(w for w,k in zip(weights,groups) if k==g) for g in GROUPS}
    return [w*SHARES[g]*len(weights)/totals[g] for w,g in zip(weights,groups)]


def base_weights(goals,groups):
    """New terrain must not change the old distance/bearing bucket frequencies."""
    from .navigation_refinement import correction_weights
    _counts(groups)
    if len(goals)!=len(groups):raise ValueError('one goal per supervision row required')
    values=[0.]*len(goals)
    for is_step in (False,True):
        indices=[i for i,g in enumerate(groups) if (g=='step')==is_step]
        weights=correction_weights([goals[i] for i in indices])
        for i,w in zip(indices,weights):values[i]=w
    return values


def reweight_step_requests(weights,groups,jumps,multiplier=4.):
    _counts(groups)
    if len(weights)!=len(groups) or len(jumps)!=len(groups) or any(type(v) is not bool for v in jumps):
        raise ValueError('step jump annotations must match supervision rows')
    if any(j and g!='step' for j,g in zip(jumps,groups)) or not any(jumps):
        raise ValueError('positive step jump requests required, with no other sources annotated')
    if type(multiplier) not in (int,float) or not 1<=multiplier<=8:
        raise ValueError('bounded step request multiplier required')
    if any(not math.isfinite(w) or w<=0 for w in weights):raise ValueError('invalid BC weights')
    before=sum(w for w,g in zip(weights,groups) if g=='step')
    scaled=[w*(multiplier if j else 1.) for w,j in zip(weights,jumps)]
    after=sum(w for w,g in zip(scaled,groups) if g=='step')
    result=[w*before/after if g=='step' else original for original,w,g in zip(weights,scaled,groups)]
    return result,dict(multiplier=multiplier,jump_requests=sum(jumps),step_labels=groups.count('step'),
        total_step_mass_before=before,total_step_mass_after=sum(w for w,g in zip(result,groups) if g=='step'),
        scope='Grounded jump-request BC only; preserve step loss share and original behavior labels')


def training_batches(groups,updates):
    import torch
    _counts(groups)
    if type(updates) is not int or not 1<=updates<=4096:raise ValueError('explicit bounded optimizer budget required')
    indices={g:[i for i,k in enumerate(groups) if k==g] for g in GROUPS}
    queues={g:[] for g in GROUPS};positions={g:0 for g in GROUPS}
    def draw(group,count):
        result=[]
        while len(result)<count:
            if positions[group]==len(queues[group]):
                queues[group]=[indices[group][i] for i in torch.randperm(len(indices[group])).tolist()]
                positions[group]=0
            take=min(count-len(result),len(queues[group])-positions[group])
            result.extend(queues[group][positions[group]:positions[group]+take]);positions[group]+=take
        return result
    return [sum((draw(g,BATCH_COUNTS[g]) for g in GROUPS),[]) for _ in range(updates)]


def _step_strata(groups,jumps):
    _counts(groups)
    if len(jumps)!=len(groups) or any(type(v) is not bool for v in jumps):
        raise ValueError('one boolean step jump flag per supervision row required')
    if any(j and g!='step' for j,g in zip(jumps,groups)):
        raise ValueError('jump strata must refer only to procedural step labels')
    strata={v:[i for i,g in enumerate(groups) if g=='step' and jumps[i]==v] for v in (False,True)}
    if not all(strata.values()):raise ValueError('both step jump and nonjump strata required')
    return strata


def stratified_step_batches(groups,jumps,updates):
    """Replace only the eight step slots with one jump and seven nonjump rows."""
    import torch
    strata=_step_strata(groups,jumps)
    # Preserve the old flat/human draws and global CPU RNG progression exactly.
    generator=torch.Generator(device='cpu');generator.set_state(torch.get_rng_state())
    batches=training_batches(groups,updates)
    queues={v:[] for v in strata};positions={v:0 for v in strata}
    def draw(positive,count):
        result=[]
        while len(result)<count:
            if positions[positive]==len(queues[positive]):
                queues[positive]=[strata[positive][i] for i in torch.randperm(
                    len(strata[positive]),generator=generator).tolist()]
                positions[positive]=0
            take=min(count-len(result),len(queues[positive])-positions[positive])
            result.extend(queues[positive][positions[positive]:positions[positive]+take]);positions[positive]+=take
        return result
    for batch in batches:
        selected=iter(draw(True,1)+draw(False,7))
        for slot,index in enumerate(batch):
            if groups[index]=='step':batch[slot]=next(selected)
    return batches


def stratified_step_importance(batch_groups,jumps,population,positive_population):
    """Correct both source and step-class sampling, retaining the same objective."""
    strata=_step_strata(batch_groups,jumps)
    original=mixture_importance(batch_groups,population)
    if Counter(batch_groups)!=BATCH_COUNTS or len(strata[True])!=1 or len(strata[False])!=7:
        raise ValueError('stratified step batches require 16 flat, 1 jump, 7 nonjump, 8 human')
    if type(positive_population) is not int or not 0<positive_population<population['step']:
        raise ValueError('positive and negative step populations must both be nonempty')
    sizes={True:positive_population,False:population['step']-positive_population}
    return [value*sizes[j]/population['step']*BATCH_COUNTS['step']/len(strata[j]) if g=='step' else value
            for value,g,j in zip(original,batch_groups,jumps)]


def preserve_step_jump_mass(weights,groups,jumps,queries,goals,multiplier=4.):
    """Alternate-goal count must not dilute the original step jump/nonjump objective."""
    from .navigation_refinement import correction_weights
    _counts(groups)
    if any(len(values)!=len(groups) for values in (weights,jumps,queries,goals)):
        raise ValueError('goal contrast annotations must match rows')
    if any(type(v) is not bool for v in jumps+queries) or any(q and g!='step' for q,g in zip(queries,groups)):
        raise ValueError('boolean procedural step query flags required')
    if any(j and g!='step' for j,g in zip(jumps,groups)) or not any(queries):
        raise ValueError('explicit step contrasts and step-only jump labels required')
    if any(not math.isfinite(w) or w<=0 for w in weights) or type(multiplier) not in (int,float) or not 1<=multiplier<=8:
        raise ValueError('positive finite weights and bounded onset multiplier required')
    originals=[i for i,g in enumerate(groups) if g=='step' and not queries[i]]
    original=correction_weights([goals[i] for i in originals])
    original=[w*(multiplier if jumps[i] else 1.) for i,w in zip(originals,original)]
    original_positive=sum(w for i,w in zip(originals,original) if jumps[i])
    fraction=original_positive/sum(original)
    if not 0<fraction<1:raise ValueError('original step labels require jump and nonjump states')
    total=sum(w for w,g in zip(weights,groups) if g=='step')
    positive=sum(w for w,g,j in zip(weights,groups,jumps) if g=='step' and j)
    negative=total-positive
    if positive<=0 or negative<=0:raise ValueError('step contrasts require both action classes')
    result=[w*(total*fraction/positive if j else total*(1-fraction)/negative) if g=='step' else w
            for w,g,j in zip(weights,groups,jumps)]
    return result,dict(original_step_labels=len(originals),original_jump_requests=sum(jumps[i] for i in originals),
        goal_query_labels=sum(queries),jump_loss_fraction=fraction,original_jump_multiplier=multiplier,
        total_step_mass_before=total,total_step_mass_after=sum(w for w,g in zip(result,groups) if g=='step'),
        scope='Preserve original observed-step jump/nonjump loss mass while adding alternative goals; flat/human weights unchanged')


def mixture_importance(batch_groups,population):
    counts=_counts(batch_groups)
    if set(population)!=set(GROUPS) or any(type(v) is not int or v<=0 for v in population.values()):
        raise ValueError('positive population count per supervision group required')
    total=sum(population.values())
    return [population[g]/total*len(batch_groups)/counts[g] for g in batch_groups]


def anchor_kl(row_kl,groups,*,flat_only=False,retained_fraction=1.):
    import torch
    _counts(groups,require_all=False)
    values=row_kl.flatten()
    if values.numel()!=len(groups):raise ValueError('anchor mask must match KL rows')
    if type(retained_fraction) not in (int,float) or not 0<retained_fraction<=1:
        raise ValueError('retained population fraction must be finite within (0,1]')
    keep=torch.tensor([g=='flat' if flat_only else g!='step' for g in groups],device=values.device)
    return values.masked_fill(~keep,0).mean()/retained_fraction
