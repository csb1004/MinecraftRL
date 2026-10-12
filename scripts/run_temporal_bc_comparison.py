"""Single-use development comparison of matched tail and temporal BC candidates."""
import argparse,json,math,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

parser=argparse.ArgumentParser()
parser.add_argument('--diagnostic-kl',action='store_true',help='Separate preregistered diagnostic; preserves failed strict gate')
args=parser.parse_args()
base=ROOT/('results/temporal-bc-diagnostic-20261008' if args.diagnostic_kl else 'results/temporal-bc-comparison-20261008')
base.mkdir(exist_ok=False)
paths={name:ROOT/('results/temporal-bc-'+kind+'-20261008/checkpoint.pt')
       for name,kind in (('parent','tail'),('candidate','temporal'))}
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
old=json.loads((ROOT/'results/recovery-curriculum-worlds-20261007/worlds.json').read_text())['worlds']
natural=json.loads((ROOT/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']['2000029']
worlds=[dict(old['eval-long-2000003'],kind='long_wall'),dict(natural,seed=2000029,split='eval',kind='natural')]
state=dict(completed=False,active=None,runs=[],errors=[],warnings=[],comparisons={},worlds=worlds,
           parent=str(paths['parent']),candidate=str(paths['candidate']),optimizer_updates=0,
           comparison='same-data/window/order/lr tail control versus last recurrent block plus tail/head BC')
frozen={str(p):digest(p) for p in (*paths.values(),embed,ROOT/'docs/research-spec.md')}
state['frozen']=frozen
state['gate_policy']='diagnostic_without_KL_limit' if args.diagnostic_kl else 'original_KL_0.03_limit'

def save():write_json(base/'experiment.json',state)

def trial(label,world,checkpoint):
    rearm_allowed()
    for path,sha in frozen.items():assert digest(path)==sha
    assert digest(world['path'])==world['sha256']
    state['active']=label;save();print(label,flush=True);log=base/(label+'.log')
    report=None;run=None
    try:
        wait_owned([sys.executable,str(ROOT/'scripts/run_rl_pilot.py'),'--seed',str(world['seed']),'--split','eval',
            '--sampling-seed',str(world['seed']),'--reward-mode','wood-potential-v4','--limit-s','180',
            '--world',world['path'],'--checkpoint',str(checkpoint),'--embed',str(embed)],log)
    finally:
        lines=log.read_text().splitlines() if log.exists() else []
        if lines:
            run=Path(lines[0])/'run'
            if (run/'summary.json').exists():
                report=json.loads((run/'summary.json').read_text())
                state['runs'].append(dict(label=label,path=str(run),world=world,checkpoint_sha256=digest(checkpoint),
                                         report=report,audited=False));save()
    assert report and report['execution_valid'] and report['worker_exit_code']==0 and report['rl_split']=='eval'
    assert not report['rl_training_eligible'] and not report['process_cleanup']['remaining_live_pids']
    release=json.loads((run/'input-release-state.json').read_text());assert release['verified'] and release['held_key_count']==0
    wsl=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src','LD_LIBRARY_PATH=/usr/lib/wsl/lib',
         'MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba','.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
    wait_owned(wsl+['scripts/audit_wood_shaping.py',linux(run)],base/(label+'-reward-audit.log'))
    wait_owned([sys.executable,str(ROOT/'scripts/audit_approach_recovery.py'),str(run)],base/(label+'-recovery-audit.log'))
    state['runs'][-1].update(audited=True,release_evidence=release);save()
    print(json.dumps(dict(label=label,steps=report['steps'],success=report['server_target_reached'])),flush=True)
    return run

save()
try:
    training={name:json.loads((p.parent/'training.json').read_text()) for name,p in paths.items()}
    for name,r in training.items():
        assert r['completed'] and digest(paths[name])==r['checkpoint_sha256']
        assert r['updates']==182 and r['learning_rate']==5e-7 and r['window_size']==8
    for key in ('parent_sha256','features_sha256','plan_sha256','order_seed'):
        assert training['parent'][key]==training['candidate'][key]
    assert training['parent']['scope']=='head_and_tail' and training['candidate']['scope']=='head_tail_last_recurrent'
    runtime=json.loads((ROOT/'results/temporal-bc-runtime-20261008-v2/report.json').read_text())
    assert runtime['completed'] and {r['sha256'] for r in runtime['checkpoints']}=={digest(p) for p in paths.values()}
    state['training_reports']={name:str(p.parent/'training.json') for name,p in paths.items()}
    state['strict_gate_passed']={name:r['runtime_comparison_eligible'] for name,r in training.items()}
    if args.diagnostic_kl:
        strict_path=ROOT/'results/temporal-bc-comparison-20261008/experiment.json'
        strict=json.loads(strict_path.read_text())
        assert strict['completed'] and not strict['runs'] and strict.get('unrun_reason') and strict['frozen']==frozen
        protocol=ROOT/'docs/temporal-bc-diagnostic-protocol-2026-10-08.md'
        state['diagnostic_protocol_sha256']=digest(protocol);state['strict_report_sha256']=digest(strict_path)
        for r in training.values():
            assert r['after']['development']['nll']<=r['before']['development']['nll']
            assert math.isfinite(r['after']['train']['reference_kl']) and r['after']['train']['reference_kl']>=-1e-6
            assert r['initial_replay_max_error']<1e-5 and r['reload_max_error']==0 and r['actor_max_abs_change']>0
        assert training['candidate']['temporal_max_abs_change']>0
        assert not training['candidate']['runtime_comparison_eligible'] and training['candidate']['after']['train']['reference_kl']>.03
    if not all(r['runtime_comparison_eligible'] for r in training.values()) and not args.diagnostic_kl:
        state['unrun_reason']='offline BC gate not met; four planned conditions unrun'
    else:
        for i,world in enumerate(worlds):
            pair={}
            for name in (('parent','candidate') if i==0 else ('candidate','parent')):
                pair[name]=trial(world['kind']+'-'+name,world,paths[name])
            comparison=compare_initial(pair['parent'],pair['candidate'])
            state['comparisons'][world['kind']]=comparison;save()
            if not (comparison['state_identical'] and comparison['pixel_identical']):
                state['warnings'].append(world['kind']+': initial mismatch; causal/time comparison invalid')
    for path,sha in frozen.items():assert digest(path)==sha
    for w in worlds:assert digest(w['path'])==w['sha256']
    state['comparison_valid']={k:v['state_identical'] and v['pixel_identical'] for k,v in state['comparisons'].items()}
    state['completed']=True
except BaseException as exc:state['errors'].append(repr(exc))
finally:
    state['active']=None;state['execution_finished']=True;save()
raise SystemExit(0 if state['completed'] else 2)
