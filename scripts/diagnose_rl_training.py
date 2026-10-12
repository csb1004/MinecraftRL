"""Read-only credit assignment and frozen-feature policy intervention.

No optimizer, gameplay, checkpoint mutation, or evaluation-data training.
Counterfactual rewards/lambda below diagnose the estimator, not policy performance.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'external/steve1'))
from gym3.types import DictType
from steve1.VPT.lib.action_head import make_action_head
from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
from minecraftrl.policies import model_kwargs
from minecraftrl.rl_core import gae
from minecraftrl.rl_opt import ppo_loss
from minecraftrl.rl_policy import guided
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from audit_rl_pilot import local


def stats(values):
    x = torch.tensor(values, dtype=torch.float64)
    return dict(count=len(values), mean=x.mean().item(), min=x.min().item(),
                median=x.median().item(), p95=x.quantile(.95).item(), max=x.max().item())


def return_stats(records):
    target = torch.tensor([r['mc_return'] for r in records], dtype=torch.float64)
    result = {}
    for name in ('before_value', 'after_value', 'gae_target'):
        prediction = torch.tensor([r[name] for r in records], dtype=torch.float64)
        result[name] = dict(mse=(prediction-target).square().mean().item(),
                            explained_variance=1-(prediction-target).var(unbiased=False).item()/target.var(unbiased=False).item())
    return result


def main():
    p = argparse.ArgumentParser();p.add_argument('batch', type=Path);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args();base = args.batch.resolve();output = args.output.resolve()
    assert output.is_relative_to(ROOT/'results') and (ROOT/'results/STOP').exists()
    output.mkdir(exist_ok=False, parents=True)
    state = json.loads((base/'experiment.json').read_text());assert state['completed']
    parent = local(state['parent_checkpoint'], ROOT);child = base/'training/checkpoint.pt'
    hashes = {str(parent):digest(parent), str(child):digest(child), str(ROOT/'docs/research-spec.md'):digest(ROOT/'docs/research-spec.md')}
    checkpoints = {name:torch.load(path, map_location='cpu') for name,path in (('before',parent),('after',child))}
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    mapping = CameraHierarchicalMapping(n_camera_bins=11)
    heads = {}
    for name,c in checkpoints.items():
        width = c['value']['weight'].shape[1]
        h = make_action_head(DictType(**mapping.get_action_space_update()), width,
                             **model_kwargs(ROOT/'weights/steve1/2x.model')[1]).cuda().eval().requires_grad_(False)
        h.load_state_dict(c['actor']);heads[name] = h
    # An independent telescoping identity for terminal Monte-Carlo GAE.
    ga, gr = gae([0.,0.,1.],[.2,.1,.4],0.,terminal=True,gamma=.9,lam=1)
    assert max(abs(a-b) for a,b in zip(gr,[.81,.9,1.])) < 1e-12
    # All four sign/clipping cases, including the negative-advantage branch.
    ratios = torch.tensor([1.5,.5,1.5,.5], requires_grad=True)
    loss = ppo_loss(ratios.log(),torch.zeros(4),torch.tensor([1.,1.,-1.,-1.]),.2)
    loss.backward();assert torch.allclose(ratios.grad,torch.tensor([0.,-.25,.25,0.]))
    assert abs(loss.item()-.15)<1e-6
    records = [];runs = [];replayed = 0;replay_max_error = 0.
    with torch.no_grad():
        for run in state['runs']:
            if run['label'].startswith('prepare-'):continue
            print('diagnose '+run['label'],flush=True)
            data = torch.load(local(run['path'],ROOT)/'trajectory.pt',map_location='cpu')
            rows = data['rows'];n = len(rows);phase = 'after' if run['label'].endswith('-after') else 'before'
            rewards = [r['reward'] for r in rows];values = [r['value'] for r in rows]
            assert data['terminal'] and sum(rewards)==1 and rewards[-1]==1
            adv, targets = gae(rewards,values,0.,terminal=True)
            no_reward, _ = gae([0.]*n,values,0.,terminal=True)
            mc_adv, mc = gae(rewards,values,0.,terminal=True,lam=1.)
            longer, _ = gae(rewards,values,0.,terminal=True,lam=.99)
            torch.manual_seed(data['seed']);torch.cuda.manual_seed_all(data['seed'])
            start = len(records);matched = 0;env_changed = 0;latent_changed = 0;first_change = None
            for i,row in enumerate(rows):
                features = row['features'].cuda()
                pd = {name:guided(head,features) for name,head in heads.items()}
                rng = torch.cuda.get_rng_state()
                samples = {'before':heads['before'].sample(pd['before'])}
                torch.cuda.set_rng_state(rng)
                samples['after'] = heads['after'].sample(pd['after'])
                same = all(torch.equal(samples[phase][k].cpu(),row['action'][k]) for k in samples[phase])
                assert same, ('saved action replay failed',run['label'],i)
                matched += same;replayed += same
                actions = {k:v.cuda() for k,v in row['action'].items()}
                logp = {name:head.logprob(actions,pd[name]).item() for name,head in heads.items()}
                error = abs(logp[phase]-row['old_logp']);replay_max_error = max(replay_max_error,error)
                assert error<2e-4, ('saved probability replay failed',run['label'],i,error)
                changed = any(not torch.equal(samples['before'][k],samples['after'][k]) for k in samples['before'])
                latent_changed += changed
                factored = {name:mapping.to_factored({k:v[:,0].cpu().numpy() for k,v in sample.items()}) for name,sample in samples.items()}
                effective = any((factored['before'][k]!=factored['after'][k]).any() for k in factored['before'])
                env_changed += bool(effective)
                if effective and first_change is None:first_change = i+1
                kl = heads['before'].kl_divergence(pd['before'],pd['after']).item()
                entropy = {name:head.entropy(pd[name]).item() for name,head in heads.items()}
                v = {name:torch.nn.functional.linear(row['features'][0:1],c['value']['weight'],c['value']['bias']).item() for name,c in checkpoints.items()}
                direct = (.99*.95)**(n-1-i)
                assert abs(adv[i]-no_reward[i]-direct)<1e-10
                assert abs(mc[i]-.99**(n-1-i))<1e-10
                records.append(dict(label=run['label'],seed=run['seed'],split=data['split'],phase=phase,step=i+1,
                    remaining=n-1-i,advantage=adv[i],no_reward_advantage=no_reward[i],direct_reward_credit=direct,
                    mc_advantage=mc_adv[i],lambda99_advantage=longer[i],mc_return=mc[i],gae_target=targets[i],
                    before_value=v['before'],after_value=v['after'],log_ratio=logp['after']-logp['before'],
                    kl_before_after=kl,entropy_before=entropy['before'],entropy_after=entropy['after'],
                    latent_action_changed=bool(changed),effective_action_changed=bool(effective)))
            subset = records[start:]
            early = [r for r in subset if r['remaining']>=100]
            runs.append(dict(label=run['label'],seed=run['seed'],steps=n,split=data['split'],phase=phase,
                replayed=matched,latent_action_changes=latent_changed,effective_action_changes=env_changed,
                first_effective_action_change=first_change,kl=stats([r['kl_before_after'] for r in subset]),
                first_value=values[0],first_mc_return=mc[0],first_gae_target=targets[0],
                first_direct_reward_credit=subset[0]['direct_reward_credit'],
                early_steps=len(early),early_raw_positive=sum(r['advantage']>0 for r in early),
                early_mc_positive=sum(r['mc_advantage']>0 for r in early),
                early_mc_sign_disagreements=sum((r['advantage']>0)!=(r['mc_advantage']>0) for r in early)))
            write_json(output/'progress.json',dict(completed_runs=len(runs),replayed=replayed,last=runs[-1]))
    training = [r for r in records if r['split']=='train']
    adv_tensor = torch.tensor([r['advantage'] for r in training], dtype=torch.float64)
    mean = adv_tensor.mean().item();std = adv_tensor.std(unbiased=False).item()
    aggregates = {}
    for split in ('train','eval'):
        group = [r for r in records if r['split']==split]
        early = [r for r in group if r['remaining']>=100]
        aggregates[split] = dict(samples=len(group),kl=stats([r['kl_before_after'] for r in group]),
            effective_action_changes=sum(r['effective_action_changed'] for r in group),
            entropy_before=statistics.mean(r['entropy_before'] for r in group),
            entropy_after=statistics.mean(r['entropy_after'] for r in group),
            value_vs_observed_discounted_return=return_stats(group),
            early_steps=len(early),early_raw_positive=sum(r['advantage']>0 for r in early),
            early_mc_positive=sum(r['mc_advantage']>0 for r in early),
            early_gae_vs_mc_sign_disagreements=sum((r['advantage']>0)!=(r['mc_advantage']>0) for r in early))
    for r in training:r['normalized_advantage'] = (r['advantage']-mean)/std
    ratio = [math.exp(r['log_ratio']) for r in training]
    a = [r['normalized_advantage'] for r in training]
    surrogate = statistics.mean(min(x*y,min(1.2,max(.8,x))*y) for x,y in zip(ratio,a))
    optimization = dict(normalized_advantage_mean=statistics.mean(a),ratio=stats(ratio),
        ratio_outside_clip=sum(x<.8 or x>1.2 for x in ratio),
        clipped_objective_saturated=sum((y>0 and x>1.2) or (y<0 and x<.8) for x,y in zip(ratio,a)),
        clipped_surrogate_before=statistics.mean(a),clipped_surrogate_after=surrogate,
        positive_advantage_mean_log_ratio=statistics.mean(r['log_ratio'] for r in training if r['normalized_advantage']>0),
        negative_advantage_mean_log_ratio=statistics.mean(r['log_ratio'] for r in training if r['normalized_advantage']<0))
    for path,h in hashes.items():assert digest(path)==h
    assert (ROOT/'results/STOP').exists()
    result = dict(verified=True,scope='read-only saved-feature and estimator interventions; not new rollout or training',
        optimizer_updates=0,gameplay_runs=0,replayed_actions=replayed,replay_logp_max_error=replay_max_error,
        mathematical_checks=['terminal lambda=1 telescopes to empirical return','PPO gradient four sign/clip branches','removing terminal reward changes raw GAE by (gamma*lambda)^distance'],
        hashes=hashes,runs=runs,aggregates=aggregates,optimization=optimization,
        temporal_weights={str(n):dict(gamma=.99**n,gae_direct=(.99*.95)**n) for n in (20,50,100,200,500)},
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write_json(output/'diagnosis.json',result)
    with (output/'step-metrics.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','hashes')}))


if __name__=='__main__':main()
