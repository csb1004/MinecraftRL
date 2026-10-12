"""Explicit experimental return targets; the original reward contract is unchanged."""
import math
from .rl_value import discounted_returns

def censored_targets(rewards,terminal,gamma,bootstrap):
    if not 0<=gamma<=1 or not 0<=bootstrap<=1:
        raise ValueError('bounded discounted goal-return required')
    target=discounted_returns(rewards,bootstrap,terminal,gamma=gamma)
    low=discounted_returns(rewards,0.,terminal,gamma=gamma)
    high=discounted_returns(rewards,1.,terminal,gamma=gamma)
    return target,low,high

def checked_training_bundle(bundle,hashes,samples):
    if (bundle.get('schema')!=1 or bundle.get('trajectory_sha256')!=hashes
        or not 0<=bundle.get('gamma',-1)<=1 or not 0<=bundle.get('lam',-1)<=1
        or set(bundle.get('value',{}))!={'weight','bias'}):
        raise ValueError('incompatible target bundle')
    for name in ('advantages','returns'):
        values=bundle.get(name,[])
        if len(values)!=samples or not all(math.isfinite(x) for x in values):
            raise ValueError('invalid '+name)
    return bundle
