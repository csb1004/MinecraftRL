"""Preregistered new-world comparison statistics; never selects runtime actions."""
import math
import statistics

def summarize_comparison(runs,seeds):
    names=('parent','control','candidate');seen=set();usable=[]
    if len(seeds)!=5 or len(set(seeds))!=5:raise ValueError('five distinct preregistered worlds required')
    for run in runs:
        key=(run['seed'],run['condition'],run['repeat'])
        if key in seen or key[0] not in seeds or key[1] not in names or key[2] not in (1,2):
            raise ValueError('duplicate or unplanned result')
        seen.add(key);report=run['report']
        if not report['execution_valid']:continue
        if report['status'] not in ('target_reached','time_limit','environment_done'):
            raise ValueError('invalid outcome status')
        success=report['server_target_reached'];seconds=report['elapsed_control_s']
        if not math.isfinite(seconds) or seconds<0 or success!=(report['status']=='target_reached'):
            raise ValueError('invalid result evidence')
        usable.append(dict(seed=key[0],condition=key[1],repeat=key[2],success=success,
            restricted_seconds=min(seconds,180.) if success else 180.,ticks=report['steps']))
    conditions={}
    for name in names:
        selected=[x for x in usable if x['condition']==name];worlds={}
        for seed in seeds:
            rows=[x for x in selected if x['seed']==seed]
            worlds[str(seed)]=dict(valid=len(rows),successes=sum(x['success'] for x in rows),
                successful_ticks=[x['ticks'] for x in rows if x['success']],
                restricted_mean_s=statistics.mean(x['restricted_seconds'] for x in rows) if rows else None)
        conditions[name]=dict(valid=len(selected),successes=sum(x['success'] for x in selected),worlds=worlds,
            restricted_mean_s=statistics.mean(x['restricted_seconds'] for x in selected) if selected else None)
    complete=len(usable)==30
    candidate=conditions['candidate'];comparators=[conditions[x] for x in ('parent','control')]
    no_regression=complete and all(candidate['worlds'][str(s)]['successes']>=c['worlds'][str(s)]['successes'] for s in seeds for c in comparators)
    faster=complete and all(candidate['restricted_mean_s']<=.9*c['restricted_mean_s'] for c in comparators)
    wins=sum(all(candidate['worlds'][str(s)]['restricted_mean_s']<c['worlds'][str(s)]['restricted_mean_s'] for c in comparators) for s in seeds) if complete else 0
    return dict(complete=complete,conditions=conditions,no_world_success_regression=bool(no_regression),
        ten_percent_faster_than_both=bool(faster),worlds_faster_than_both=wins,
        candidate_gate_passed=bool(complete and no_regression and faster and wins>=3))
