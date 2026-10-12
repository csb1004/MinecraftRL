"""Isolated value calibration; never selects or updates policy actions."""
import math


def calibration_passes(scores,improved_seeds,seed_count):
    if seed_count<2 or not 0<=improved_seeds<=seed_count:raise ValueError('invalid fold count')
    values=[scores[k]['mse'] for k in ('inherited','constant','calibrated')]+[scores[k]['early_mse'] for k in ('inherited','calibrated')]
    if any(not math.isfinite(v) or v<0 for v in values):return False
    return (scores['calibrated']['mse']<=.9*min(scores['inherited']['mse'],scores['constant']['mse'])
            and improved_seeds>=math.ceil(.75*seed_count)
            and scores['calibrated']['early_mse']<scores['inherited']['early_mse'])


def discounted_returns(rewards,bootstrap,terminal,*,gamma=.99):
    if not rewards or not 0<=gamma<=1 or not math.isfinite(bootstrap) or any(not math.isfinite(r) for r in rewards):
        raise ValueError('invalid return inputs')
    result=[];following=0. if terminal else float(bootstrap)
    for reward in reversed(rewards):
        following=float(reward)+gamma*following;result.append(following)
    return result[::-1]


def seed_folds(episodes):
    if not episodes or any(e['split']!='train' for e in episodes):
        raise ValueError('only training episodes may calibrate value')
    seeds=sorted({e['seed'] for e in episodes})
    if len(seeds)<2:raise ValueError('multiple training seeds required')
    for seed in seeds:
        yield seed,[i for i,e in enumerate(episodes) if e['seed']!=seed],[i for i,e in enumerate(episodes) if e['seed']==seed]


def fit_residual_value(features,targets,weight,bias,*,ridge=1.):
    import torch
    if not math.isfinite(ridge) or ridge<=0:raise ValueError('positive ridge required')
    x=features.detach().to(dtype=torch.float64,device='cpu')
    y=targets.detach().to(dtype=torch.float64,device='cpu').flatten()
    w=weight.detach().to(dtype=torch.float64,device='cpu');b=bias.detach().to(dtype=torch.float64,device='cpu')
    if x.ndim!=2 or x.shape[0]!=len(y) or len(y)<2 or w.shape!=(1,x.shape[1]) or b.shape!=(1,):
        raise ValueError('inconsistent value fitting shapes')
    if not all(torch.isfinite(v).all() for v in (x,y,w,b)):raise ValueError('nonfinite value inputs')
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6)
    normalized=(x-mean)/scale
    residual=y-torch.nn.functional.linear(x,w,b).flatten();offset=residual.mean()
    gram=normalized.T@normalized/len(y)
    gram.diagonal().add_(ridge)
    delta=torch.linalg.solve(gram,normalized.T@(residual-offset)/len(y))
    correction=delta/scale
    fitted=dict(weight=w+correction.unsqueeze(0),bias=b+offset-correction@mean)
    if not all(torch.isfinite(v).all() for v in fitted.values()):raise ValueError('nonfinite fitted value')
    return fitted
