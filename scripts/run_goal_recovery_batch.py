"""One-use, frozen corner-first paired goal-conditioning development diagnostic."""
import json
import argparse
import shutil
import sys
from pathlib import Path

from run_rl_batch import ROOT, LINUX, compare_initial, linux, rearm_allowed, wait_owned
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


base = ROOT/'results/goal-recovery-comparison-20261009'
p = argparse.ArgumentParser(); resume = p.add_mutually_exclusive_group()
resume.add_argument('--resume-audit',action='store_true')
resume.add_argument('--resume-startup-fix',action='store_true')
args = p.parse_args()
if not (args.resume_audit or args.resume_startup_fix): base.mkdir(exist_ok=False)
checkpoint = ROOT/'results/onset-coverage-training-20261009/checkpoint.pt'
assert digest(checkpoint) == '473142f1283521eb4417d3b4c6f568af27ad4b1a83113c4b506c9f2f6da153dd'
embed = ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
conditions = {k:ROOT/'results/goal-conditioning-diagnosis-20261009'/f'{k}_9711.npy' for k in ('wood','explore')}
worlds = read(ROOT/'results/diverse-teacher-worlds-20261008/worlds.json')['worlds']
natural = read(ROOT/'results/rl-batch-20261005T040215634778Z/batch.json')['worlds']['2000029']
worlds = dict(corner=dict(worlds['development-corner'],kind='corner'),
              far=dict(worlds['development-far'],kind='far'),
              natural=dict(natural,seed=2000029,kind='natural'))
files = [checkpoint,embed,*conditions.values(),Path(__file__),ROOT/'scripts/audit_goal_recovery_run.py',
    ROOT/'scripts/run_goal_recovery_pilot.py',ROOT/'src/minecraftrl/goal_recovery.py',
    ROOT/'src/minecraftrl/cli.py',ROOT/'src/minecraftrl/runtime.py',ROOT/'src/minecraftrl/rl_policy.py',
    ROOT/'src/minecraftrl/minerl_runtime.py',
    ROOT/'src/minecraftrl/rl_control.py',ROOT/'src/minecraftrl/stall_detection.py',
    ROOT/'docs/goal-recovery-pilot-protocol-2026-10-09.md',ROOT/'docs/research-spec.md']
frozen = {str(p):digest(p) for p in files}
state = dict(completed=False,active=None,runs=[],unrun=[],errors=[],comparisons={},frozen=frozen,
    worlds=worlds,planned_conditions=6,optimizer_updates=0,llm_calls=0,colab_connections=0,
    default_promoted=False,scope='Privileged graph-triggered goal-conditioning diagnostic; neural actions only')
if args.resume_audit:
    old = read(base/'experiment.json')
    assert not old['completed'] and old['active'] is None and len(old['runs']) == 1
    assert old['runs'][0]['report']['execution_valid'] and not old['runs'][0]['audited']
    archived = ROOT/'results/goal-recovery-implementation-20261009/audit-fix-originals'
    assert read(archived/'experiment.json') == old
    changed = []
    for path,sha in old['frozen'].items():
        if digest(path) != sha:
            assert Path(path).name in ('audit_goal_recovery_run.py','run_goal_recovery_batch.py')
            assert digest(archived/Path(path).name) == sha
            changed.append(dict(path=path,previous_sha256=sha,new_sha256=digest(path),
                                original=str(archived/Path(path).name)))
    assert len(changed) == 2
    state = old
    state.update(previous_errors=old['errors'],errors=[],frozen=frozen,execution_finished=False,
        audit_fix=dict(reason='JSON list versus native tuple comparison at transition 1411; reward values unchanged',
                       sources=changed,original_game_reused=True))
elif args.resume_startup_fix:
    old = read(base/'experiment.json')
    archived = ROOT/'results/goal-recovery-implementation-20261009/startup-fix-originals'
    assert read(archived/'experiment.json') == old
    assert not old['completed'] and old['active'] is None and len(old['runs']) == 2
    good,failed = old['runs']
    assert good['label']=='corner-wood' and good['audited']
    assert failed['label']=='corner-explore' and not failed['audited'] and failed['report']['steps']==0
    assert failed['report']['inference_ms']['count']==0 and not failed['report']['process_cleanup']['remaining_live_pids']
    assert 'process_watcher' in failed['report']['error']
    changes = []
    for path,sha in old['frozen'].items():
        if digest(path) != sha:
            assert Path(path).name in ('cli.py','run_goal_recovery_batch.py')
            assert digest(archived/Path(path).name) == sha
            changes.append(dict(path=path,previous_sha256=sha,new_sha256=digest(path)))
    changes.append(dict(path=str(ROOT/'src/minecraftrl/minerl_runtime.py'),
        previous_sha256=digest(archived/'minerl_runtime.py'),new_sha256=digest(ROOT/'src/minecraftrl/minerl_runtime.py')))
    state = old
    state.update(errors=[],previous_startup_errors=old['errors'],frozen=frozen,execution_finished=False,
        runs=[good],technical_attempts=[dict(failed,counts_in_performance=False)],
        startup_fix=dict(reason='Upstream stale PID file collision; per-episode watcher cwd',sources=changes,
                         original_completed_game_reused=True,retry_limit=1))


def save(): write_json(base/'experiment.json',state)


def trial(kind,condition,seed):
    rearm_allowed()
    assert shutil.disk_usage(ROOT).free > 6*2**30, 'Insufficient disk headroom for preserved game logs'
    for path,sha in frozen.items(): assert digest(path) == sha, path
    world = worlds[kind]; assert digest(world['path']) == world['sha256']
    label = kind+'-'+condition; state['active'] = label; save(); print(label,flush=True)
    retry = any(t['label']==label for t in state.get('technical_attempts',[]))
    stem = label+('-retry-startup-1' if retry else '')
    log = base/(stem+'.log'); report = run = None
    existing = next((r for r in state['runs'] if r['label']==label),None)
    if existing:
        run,report = Path(existing['path']),existing['report']
        assert read(run/'summary.json') == report
        if existing['audited']: return run,existing['audit']
    else:
        assert not log.exists(), 'Never repeat a previously started game'
        try:
            wait_owned([sys.executable,str(ROOT/'scripts/run_goal_recovery_pilot.py'),
                '--seed',str(world['seed']),'--sampling-seed',str(seed),'--checkpoint',str(checkpoint),
                '--embed',str(embed),'--recovery-embed',str(conditions[condition]),'--world',world['path']],log)
        finally:
            lines = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
            if lines:
                run = Path(lines[0])/'run'
                if (run/'summary.json').exists():
                    report = read(run/'summary.json')
                    state['runs'].append(dict(label=label,kind=kind,condition=condition,sampling_seed=seed,
                        path=str(run),report=report,audited=False)); save()
    assert report and report['execution_valid'] and report['worker_exit_code'] == 0
    wsl = ['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src',
        'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+LINUX+'/.runtime/mamba',
        '.runtime/bin/micromamba','run','-p',LINUX+'/.runtime/steve1','python']
    audit_log = base/(stem+('-audit-after-format-fix.log' if existing else '-audit.log'))
    assert not audit_log.exists()
    wait_owned(wsl+['scripts/audit_goal_recovery_run.py',linux(run),linux(checkpoint)],audit_log)
    audit = read(run/'goal-recovery-audit.json')
    state['runs'][-1].update(audited=True,audit=audit); save()
    print(json.dumps(dict(label=label,steps=audit['steps'],wood=audit['log_gain'],
        intervention_steps=audit['intervention_steps'],expansion_eligible=audit['expansion_eligible'])),flush=True)
    return run,audit


def pair(kind,order,seed):
    runs = {condition:trial(kind,condition,seed) for condition in order}
    comparison = compare_initial(runs['wood'][0],runs['explore'][0])
    audits = [runs[c][1] for c in ('wood','explore')]
    switches = [next((s for s in a['switches'] if s['mode']=='intervention' and s['committed']),None) for a in audits]
    comparable = all(switches)
    comparison['pre_intervention_prefix_identical'] = comparable and audits[0]['prefix_action_frame_sha256'] == audits[1]['prefix_action_frame_sha256']
    comparison['trigger_identical'] = comparable and all(switches[0][key] == switches[1][key]
        for key in ('after_step','server_before','memory_sha256','input_image_sha256'))
    comparison['text_only_causal_comparison_valid'] = all(comparison[k] for k in
        ('state_identical','pixel_identical','pre_intervention_prefix_identical','trigger_identical'))
    state['comparisons'][kind] = comparison; save()
    return runs['explore'][1]


save()
try:
    corner = pair('corner',('wood','explore'),9721)
    state['expansion_admitted'] = corner['expansion_eligible']; save()
    if state['expansion_admitted']:
        pair('far',('explore','wood'),9722)
        pair('natural',('wood','explore'),9723)
    else:
        state['unrun'] = [dict(kind=k,condition=c,reason='corner movement/yaw admission not met')
                          for k in ('far','natural') for c in ('wood','explore')]
    for path,sha in frozen.items(): assert digest(path) == sha
    assert len(state['runs'])+len(state['unrun']) == state['planned_conditions']
    state['completed'] = True
except BaseException as exc:
    state['errors'].append(repr(exc))
finally:
    state.update(active=None,execution_finished=True); save()
raise SystemExit(0 if state['completed'] else 2)
