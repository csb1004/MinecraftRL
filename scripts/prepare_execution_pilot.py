"""Freeze E2 cases before policy collection; preparation creates no policy samples."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from minecraftrl.goal_spec import compile_spec,request

NAME='execution-contract-e2-20261103'
TRAIN_SEEDS=(81113001,81113003)
DEV_SEEDS=(81112001,81112003,81112007,81112009,81112013)

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def goal(ref,quantity=1,*,effect='acquire',previous=None,preserve=None):
    return dict(id=ref+('-hold' if effect=='possess' else '-gain'),target=ref,quantity=quantity,
        priority=80,depends_on=[] if previous is None else [previous],requires=[],effect=effect,
        preserve=[] if preserve is None else [dict(target=preserve,quantity=1 if effect=='acquire' else 2)])

def programs():
    p={}
    for effect in ('acquire','possess'):
        for first,second in [('wood','dirt'),('dirt','wood')]:
            a=goal(first,1 if effect=='acquire' else 2,effect=effect)
            b=goal(second,2 if effect=='acquire' else 1,effect=effect,previous=a['id'],preserve=first)
            p['train-'+effect+'-'+first]=[a,b]
    p['wood3']=[goal('wood',3)]
    p['dirt3-hold']=[goal('dirt',3,effect='possess')]
    for a,b in [('wood','dirt'),('dirt','wood')]:
        first=goal(a);second=goal(b,previous=first['id'],preserve=a)
        p[a+'-then-'+b]=[first,second]
    first=goal('wood');p['wood-then-held']=[first,goal('wood',effect='possess',previous=first['id'],preserve='wood')]
    # The successor asks only for the one unit just acquired.
    p['wood-then-held'][1]['preserve'][0]['quantity']=1
    p['pickup-wood']=[goal('wood')];p['pickup-dirt']=[goal('dirt')];p['stop']=[goal('wood',64)]
    return {key:dict(schema='goal_spec_v2',goals=goals) for key,goals in p.items()}

def prepare(root,batch):
    if (batch/'preparation-plan.json').exists():raise ValueError('E2 inputs already prepared')
    bank=root/'results/common-recovery-r1-20261101/goal-bank'
    shutil.copytree(bank,batch/'goal-bank')
    (batch/'programs').mkdir();(batch/'worlds').mkdir()
    for name,spec in programs().items():
        compile_spec(spec);write(batch/'programs'/(name+'.json'),spec)
    write(batch/'authoring-request-v2.json',request(2))
    plan=dict(protocol='execution_contract_e2_v1',preparation_seeds=list(TRAIN_SEEDS),
        development_seeds=list(DEV_SEEDS),policy_runs=26,performance_stop_probe=False,
        models=['initial','plain','calibrated','retained'],train_runs=4,retention_runs=2,
        max_control_s=180,automatic_promotion=False,API_calls=0,
        training=dict(epochs=2,actor_lr=.0001,value_lr=.001,gamma=.999,gae_lambda=1.,
            shaping_alpha=.2,calibration_epochs=10,retention_kl=.1,
            balance='equal observed target/effect mass; no extra row visits'),
        claim='small descriptive development test; known development worlds, not held-out generalization',
        no_success_seeking_reruns=True)
    write(batch/'preparation-plan.json',plan)
    print(json.dumps(plan,indent=2))

def finalize(root,batch):
    plan=read(batch/'preparation-plan.json')
    for seed in TRAIN_SEEDS:
        src=batch/('prepare-'+str(seed))/'initial-world.zip'
        meta=read(src.with_suffix('.json'))
        if meta['seed']!=seed or meta['sha256']!=sha(src):raise ValueError('new prepared world mismatch')
        out=batch/'worlds'/('natural-'+str(seed)+'.zip')
        if out.exists():raise ValueError('do not replace prepared inputs')
        shutil.copy2(src,out);meta['experimental_split']='train'
        meta['E2_role_assignment']='new train-only copy; neutral preparation not a performance sample'
        write(out.with_suffix('.json'),meta)
    runs=[]
    def case(name,world,program,role,data_role,seed,sampling_seed,model='initial',task='collection',case_id=None):
        worldpath=root/world;meta=read(worldpath.with_suffix('.json'))
        if meta['seed']!=seed or meta['experimental_split']!=role or meta['sha256']!=sha(worldpath):
            raise ValueError('world role/identity mismatch')
        runs.append(dict(name=name,case_id=case_id or name,world=world,world_sha256=meta['sha256'],
            program=program,role=role,data_role=data_role,seed=seed,sampling_seed=sampling_seed,
            model=model,candidate=model!='initial',checkpoint=None if model=='initial' else
            f'results/{NAME}/training-{model}/checkpoint.pt',variant='nonlinear-on',task=task,
            action_budget=400 if task=='pickup' else 1200,limit_s=180))
    for seed,effect in zip(TRAIN_SEEDS,('acquire','possess')):
        for index,first in enumerate(('wood','dirt')):
            name=f'train-{seed}-{first}'
            case(name,f'results/{NAME}/worlds/natural-{seed}.zip','train-'+effect+'-'+first,
                'train','PPO',seed,seed+index)
    for index,(ref,world) in enumerate([('wood','train-r2-wood-far'),('dirt','train-r2-dirt-turned')]):
        case('retention-'+ref,'results/common-recovery-r1-20261101/worlds/'+world+'.zip',
            'pickup-'+ref,'train','official_KL_only',81111001,81113901+index,task='pickup')
    devprograms=('wood3','dirt3-hold','wood-then-dirt','dirt-then-wood','wood-then-held')
    models=plan['models']
    for index,(seed,program) in enumerate(zip(DEV_SEEDS,devprograms)):
        order=models[index%4:]+models[:index%4]
        for model in order:
            case('dev-'+program+'-'+model,f'results/common-recovery-r1-20261101/worlds/natural-{seed}.zip',
                program,'development','evaluation_only',seed,seed+400,model=model,case_id='dev-'+program)
    assert len(runs)==plan['policy_runs']
    write(batch/'plan.json',dict(plan,runs=runs))
    print('Frozen 26 policy cases; no game or training performed by this helper.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','finalize'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();batch=root/'results'/NAME
    (prepare if a.operation=='prepare' else finalize)(root,batch)
