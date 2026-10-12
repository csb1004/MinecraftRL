"""One-use parent versus masked-BC student development diagnosis; no teacher control."""
import json,math,sys
from pathlib import Path
from run_rl_batch import ROOT,LINUX,linux,rearm_allowed,wait_owned,compare_initial
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

base=ROOT/'results/masked-correction-comparison-20261009';base.mkdir(exist_ok=False)
paths=dict(parent=ROOT/'results/temporal-bc-temporal-20261008/checkpoint.pt',
           candidate=ROOT/'results/masked-correction-training-20261009/checkpoint.pt')
embed=ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
old=json.loads((ROOT/'results/recovery-curriculum-worlds-20261007/worlds.json').read_text())['worlds']
natural=json.loads((ROOT/'results/rl-batch-20261005T040215634778Z/batch.json').read_text())['worlds']['2000029']
worlds=[dict(old['eval-long-2000003'],kind='long_wall'),dict(natural,seed=2000029,split='eval',kind='natural')]
protocol=ROOT/'docs/masked-correction-bc-protocol-2026-10-09.md'
state=dict(completed=False,active=None,runs=[],errors=[],warnings=[],comparisons={},worlds=worlds,
    parent=str(paths['parent']),candidate=str(paths['candidate']),optimizer_updates=0,
    gate_policy='preregistered_diagnostic_without_KL_limit',protocol_sha256=digest(protocol),
    comparison='same temporal parent before/after one teacher-only masked BC pass; no scripted assistance')
frozen={str(p):digest(p) for p in (*paths.values(),embed,protocol,ROOT/'docs/research-spec.md')}
state['frozen']=frozen
def save():write_json(base/'experiment.json',state)

def trial(label,world,checkpoint):
    rearm_allowed()
    for path,sha in frozen.items():assert digest(path)==sha
    assert digest(world['path'])==world['sha256']
    state['active']=label;save();print(label,flush=True);log=base/(label+'.log');report=run=None
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
    assert report['controller']=='steve1_head_rl' and not report['rl_training_eligible']
    assert not report['process_cleanup']['remaining_live_pids']
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
    reports={k:json.loads((p.parent/'training.json').read_text()) for k,p in paths.items()}
    r=reports['candidate']
    assert r['completed'] and r['checkpoint_sha256']==digest(paths['candidate'])
    assert r['parent_sha256']==digest(paths['parent']) and r['updates']==r['planned_updates']==130
    assert r['masked_correction'] and r['label_counts']==dict(train=1005,development=619)
    assert r['learning_rate']==5e-7 and r['window_size']==8
    runtime=json.loads((ROOT/'results/masked-correction-runtime-20261009/report.json').read_text())
    assert runtime['completed'] and {x['sha256'] for x in runtime['checkpoints']}=={digest(p) for p in paths.values()}
    state['strict_gate_passed']={k:v['runtime_comparison_eligible'] for k,v in reports.items()}
    eligible=(r['after']['development']['nll']<=r['before']['development']['nll'] and
        r['initial_replay_max_error']<1e-5 and r['reload_max_error']==0 and r['temporal_max_abs_change']>0 and
        all(math.isfinite(r['after'][s][k]) for s in ('train','development') for k in ('nll','reference_kl','all_frame_kl','context_kl')))
    if not eligible:
        state['unrun_reason']='preregistered masked BC diagnostic admission not met; four conditions unrun'
    else:
        for i,world in enumerate(worlds):
            pair={}
            for name in (('parent','candidate') if i==0 else ('candidate','parent')):
                pair[name]=trial(world['kind']+'-'+name,world,paths[name])
            comparison=compare_initial(pair['parent'],pair['candidate']);state['comparisons'][world['kind']]=comparison
            if not (comparison['state_identical'] and comparison['pixel_identical']):
                state['warnings'].append(world['kind']+': initial mismatch; causal/time comparison invalid')
            save()
    for path,sha in frozen.items():assert digest(path)==sha
    for world in worlds:assert digest(world['path'])==world['sha256']
    state['comparison_valid']={k:v['state_identical'] and v['pixel_identical'] for k,v in state['comparisons'].items()}
    state['completed']=True
except BaseException as exc:state['errors'].append(repr(exc))
finally:state.update(active=None,execution_finished=True);save()
raise SystemExit(0 if state['completed'] else 2)
