"""Consolidate completed local evidence; never dispatches environment actions."""
import argparse,datetime,json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from run_rl_batch import compare_initial

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=root/'results/recovery-curriculum-20261007');p.add_argument('--test-prefix',default='recovery-curriculum');args=p.parse_args();base=args.batch
state=json.loads((base/'experiment.json').read_text())
assert state['completed'] and not state['errors'] and state['active'] is None
assert (root/'results/STOP').exists()
assert digest(state['parent'])==state['parent_sha256']
assert digest(root/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy')==state['embed_sha256']
jar_hashes={
    '.runtime/world-save/58b912b13fbd74933c510c70ba14f9bc911266ce97eb8bc02196422f96e2fc49.jar':'afeb88a9dd572bad19f25cf8d0a7346c08ef17ab5993e9de5b9da1ebee40688c',
    '.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar':'f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48'}
for path,sha in jar_hashes.items():assert digest(root/path)==sha,path
assert digest(root/'docs/research-spec.md')==state['research_spec_sha256']=='4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad'
for path,sha in state['source_sha256'].items():assert digest(root/path)==sha,path
assert digest(root/'scripts/train_rl_heads.py')==state['trainer_sha256']
for world in state['worlds'].values():
    assert digest(world['path'])==world['sha256']
    manifest=json.loads(Path(world['path']).with_suffix('.json').read_text())
    assert digest(manifest['source'])==manifest['source_sha256']
original=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in original['training']['trajectories']:
    path=str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/')
    assert digest(path)==item['sha256']
prior_curriculum_preserved=0
if base.name=='recovery-gap-20261007':
    prior=json.loads((root/'results/recovery-curriculum-20261007/final-verification.json').read_text())
    for row in prior['runs']:
        assert digest(Path(row['path'])/'trajectory.pt')==row['trajectory_sha256']
        prior_curriculum_preserved+=1
routes=json.loads((base/'route-audit.json').read_text())['rows']
assert len(routes)==len(state['runs'])
rows=[]
for record,route in zip(state['runs'],routes):
    run=Path(record['path']);summary=json.loads((run/'summary.json').read_text())
    shaping=json.loads((run/'shaping-audit.json').read_text())
    recovery=json.loads((run/'recovery-audit.json').read_text())
    assert record['audited'] and shaping['verified'] and recovery['verified']
    assert route['label']==record['label'] and route['steps']==summary['steps']
    assert digest(record['checkpoint'])==record['checkpoint_sha256']
    assert summary['execution_valid'] and summary['worker_exit_code']==0
    assert summary['rl_training_eligible']==(record['world']['split']=='train')
    assert shaping['no_policy_action_after_stop'] and shaping['one_server_tick_per_action']
    assert not summary['process_cleanup']['remaining_live_pids']
    assert summary['release']=='neutral_command_sent'
    if 'release_evidence' in record:
        assert record['release_evidence']['verified'] and record['release_evidence']['held_key_count']==0
    rows.append(dict(label=record['label'],path=str(run),split=record['world']['split'],
        status=summary['status'],steps=summary['steps'],success=summary['server_target_reached'],
        seconds=summary['elapsed_control_s'],inference_ms=summary['inference_ms'],
        control_cycle_ms=summary['control_cycle_ms'],gpu=summary['gpu'],device_vram=summary['device_vram'],
        stages=shaping['stages'],recovery_active_snapshots=recovery['active_snapshots'],
        route=route,trajectory_sha256=digest(run/'trajectory.pt'),
        shaping_audit_sha256=digest(run/'shaping-audit.json'),recovery_audit_sha256=digest(run/'recovery-audit.json')))
for training in state['training']:
    report=training['report'];assert report['completed'] and report['optimizer_state_reloaded']
    assert report['checkpoint_reload_logp_error']==0
    assert report['behavior_logp_max_error']<=2e-4
    assert digest(Path(training['path'])/'checkpoint.pt')==report['checkpoint_sha256']
if state['accepted_checkpoint']:
    accepted=state['accepted_checkpoint'];assert digest(accepted['path'])==accepted['sha256']
    assert any(t['report']['checkpoint_sha256']==accepted['sha256'] and t['report']['runtime_comparison_eligible'] for t in state['training'])
tests={}
for name in ('windows','wsl','fixture'):
    file=root/'results'/(f'{args.test_prefix}-fixture-tests.txt' if name=='fixture' else f'{args.test_prefix}-tests-{name}.txt')
    raw=file.read_bytes();log=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    count=re.search(r'Ran (\d+) tests in',log);skip=re.search(r'^OK \(skipped=(\d+)\)',log,re.M)
    assert count and re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M)
    tests[name]=dict(run=int(count[1]),skipped=int(skip[1]) if skip else 0,sha256=digest(file))
processes=json.loads((base/'process-check.json').read_text(encoding='utf-8-sig'))
assert not processes['windows'] and not processes['wsl'] and processes['stop_latched']
context=json.loads((base/'context-diagnosis.json').read_text())
assert context['evaluated']==bool(state['accepted_checkpoint'])
repeat_comparisons={}
by_label={r['label']:r for r in state['runs']}
for label,record in by_label.items():
    if label.startswith('collect-') and label.endswith('-2'):
        first=by_label[label[:-1]+'1']
        repeat_comparisons[label]=compare_initial(Path(first['path']),Path(record['path']))
write_json(base/'final-verification.json',dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    runs=rows,actual_transitions=sum(r['steps'] for r in rows),optimizer_updates=sum(t['report']['updates'] for t in state['training']),
    splits={split:dict(runs=sum(r['split']==split for r in rows),successes=sum(r['split']==split and r['success'] for r in rows),
                      transitions=sum(r['steps'] for r in rows if r['split']==split)) for split in ('train','eval')},
    accepted_checkpoint=state['accepted_checkpoint'],curriculum_stop=state.get('curriculum_stop'),repeat_comparisons=repeat_comparisons,
    evaluation_unrun_reason=state.get('evaluation_unrun_reason'),tests=tests,
    preserved_original_training_trajectories=len(original['training']['trajectories']),preserved_prior_curriculum_trajectories=prior_curriculum_preserved,
    original_worlds_and_parent_preserved=True,source_sha256=state['source_sha256'],trainer_sha256=state['trainer_sha256'],
    research_spec_sha256=state['research_spec_sha256'],processes=processes,jar_sha256=jar_hashes,
    default_policy_promoted=False,final_seeds_unrun=state['final_seeds_unrun'],
    scope='local WSL server/inventory evidence; collection gate completion is not learning success'))
print(json.dumps(dict(verified=True,runs=len(rows),steps=sum(r['steps'] for r in rows),
    successes=sum(r['success'] for r in rows),optimizer_updates=sum(t['report']['updates'] for t in state['training']))))
