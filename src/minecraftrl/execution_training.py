"""E2 calibration and retention roles; historical anchors never become PPO rows."""
from collections import Counter


def objective_weights(rows):
    """Equal total influence per observed target/effect, without extra PPO visits."""
    if not rows:raise ValueError('nonempty verified PPO rows required')
    keys=[r['execution_request']['target_ref']+'/'+r['execution_request']['effect'] for r in rows]
    counts=Counter(keys);weights={k:len(rows)/(len(counts)*n) for k,n in counts.items()}
    return [weights[k] for k in keys],dict(method='equal_observed_target_effect_mass',
        groups={k:dict(rows=n,weight=weights[k]) for k,n in sorted(counts.items())})


def check_retention_source(data, summary, behavior, *, source, ppo_sources):
    if (source in ppo_sources or data.get('scope')!='execution_contract_e2'
            or data.get('split')!='train' or data.get('eligible') is not True
            or summary.get('execution_valid') is not True or summary.get('worker_exit_code')!=0
            or data.get('status') not in ('target_reached','time_limit','environment_done')
            or data.get('status')!=summary.get('status')):
        raise ValueError('separate valid train-only retention source required')
    for key in ('base_model_sha256','context_fields','guidance'):
        if data.get(key)!=behavior.get(key):raise ValueError('retention identity mismatch: '+key)


def calibrate_value(value, features, targets, *, epochs, check_stop, weights=None):
    import torch
    if type(epochs) is not int or not 0<=epochs<=10:
        raise ValueError('critic calibration epochs must be 0 to 10')
    if features.shape[1]!=len(targets) or not len(targets):raise ValueError('invalid calibration data')
    if weights is None:weights=torch.ones_like(targets)
    def mse():
        with torch.no_grad():return float(((value(features).flatten()-targets).square()*weights).mean())
    before=mse();updates=0
    optimizer=torch.optim.Adam(value.parameters(),lr=.001)
    generator=torch.Generator(device=features.device);generator.manual_seed(113003)
    for _ in range(epochs):
        order=torch.randperm(len(targets),device=features.device,generator=generator)
        for start in range(0,len(targets),32):
            check_stop();idx=order[start:start+32]
            loss=((value(features[:,idx]).flatten()-targets[idx]).square()*weights[idx]).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite calibration')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
    return dict(epochs=epochs,updates=updates,mse_before=before,mse_after=mse(),policy_updates=0,
                target_source='fixed rollout returns; original rollout values preserved')
