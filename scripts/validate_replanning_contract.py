"""One-use archived-stall contract audit. Proposals are explicit synthetic fixtures."""
import argparse
import copy
import json
from pathlib import Path

from minecraftrl.replanning import PlanHistory, ReplanGuard, collection_capability, make_snapshot
from minecraftrl.stall_replay import file_sha256, log_count
from replay_stall_detection import validate_manifest


def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def local(root, path):
    value = str(path).replace('\\','/')
    if '/MinecraftRL/' in value:
        value = value.split('/MinecraftRL/',1)[1]
    result = (root/value).resolve()
    if root not in result.parents:
        raise ValueError('source outside project')
    return result


def run(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if root not in output.parents:
        raise ValueError('output outside project')
    manifest_path = root/'configs/graph-stall-replay-20261008.json'
    report_path = root/'results/graph-stall-replay-20261008-v3/report.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    source = json.loads(report_path.read_text(encoding='utf-8'))
    assert source['completed'] and source['error'] is None
    assert file_sha256(manifest_path) == source['manifest_sha256']
    output.mkdir(exist_ok=False)
    (output/'histories').mkdir(); (output/'cases').mkdir()
    frozen = {str(p.relative_to(root)):file_sha256(p) for p in
              (manifest_path,report_path,root/'results/STOP',root/'docs/research-spec.md')}
    specs = {r['id']:r for r in manifest['runs']}
    result = dict(completed=False,error=None,mode='offline_synthetic_proposal_contract',
                  new_game_actions=0,optimizer_updates=0,llm_calls=0,events=[],cases=[],
                  no_stall_reference_runs=[],source_sha256=frozen,
                  implementation_sha256={p:file_sha256(root/p) for p in
                      ('src/minecraftrl/replanning.py','scripts/validate_replanning_contract.py')})
    try:
        for row in source['runs']:
            spec = specs[row['id']]
            if not row['alarms']:
                result['no_stall_reference_runs'].append(row['id'])
                continue
            run_dir = local(root,spec['run'])
            for name,sha in spec['sha256'].items():
                assert file_sha256(run_dir/name) == sha
            summary = json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
            initial = json.loads((run_dir/'initial-state.json').read_text(encoding='utf-8'))
            assert summary['goal'] == 'collect wood' and summary['controller'] == 'steve1_head_rl'
            condition = summary['condition']
            checkpoint = local(root,condition['checkpoint'])
            checkpoint_sha = file_sha256(checkpoint)
            frozen[str(checkpoint.relative_to(root))] = checkpoint_sha
            policy_ref = checkpoint_sha + ':' + condition['embedding_sha256'] + ':' + str(condition['cond_scale'])
            capabilities = {'collect_wood':collection_capability('wood',policy_ref)}
            history = PlanHistory(output/'histories'/(row['id']+'.json'))
            guard = ReplanGuard(history,capabilities)
            alarms = {a['step']:a for a in row['alarms']}
            started = None
            with (run_dir/'events.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    event = json.loads(line)
                    if event['event'] == 'episode_started':
                        started = event['monotonic_s']
                    if event['event'] != 'transition' or event['step'] not in alarms:
                        continue
                    assert started is not None
                    alarm = alarms[event['step']]
                    observed = event['info']['workshop']
                    event_id = row['id']+'-stall-'+str(event['step'])
                    snapshot = make_snapshot(observed,world_id=initial['initial_world']['sha256'],
                        episode_id=run_dir.parent.name,goal='wood',task=None,generation=None,
                        sequence=event['step'],elapsed_s=event['monotonic_s']-started,status='running',
                        release_sequence=None,held_key_count=None,baseline_count=log_count(initial['server']))
                    evidence = dict(alarm,event_id=event_id)
                    guard.record_failure(snapshot,evidence,'collect_wood')
                    request = guard.request(snapshot,evidence)
                    payload = guard.planner_input(request['request_id'])
                    assert not payload['call_model'] and payload['blocked_reason'] == 'no_available_alternative'
                    assert snapshot['task'] is None and snapshot['goal_ref']['kind'] == 'episode'
                    assert not {'barrier_material','curriculum','route','world_time'} & set(snapshot['facts'])
                    result['events'].append(dict(id=event_id,source_run=spec['run'],
                        events_sha256=spec['sha256']['events.jsonl'],step=event['step'],snapshot=snapshot,
                        request_id=request['request_id'],model_call_skipped=True,
                        blocker=payload['blocked_reason'],recorded_held_keys=observed['held_key_count'],
                        authority='archived observation; no recorded task/revision/generation/release receipt'))
                    base = dict(request_id=request['request_id'],skill_id='collect_wood',arguments={},
                                budget_s=30.,budget_ticks=200,evidence_ids=[event_id],reason='fixture proposal')
                    variants = (
                        ('same_method',{},None,0.,'same_failed_attempt'),
                        ('unsupported_detour',{'skill_id':'detour_right'},None,0.,'unsupported_skill'),
                        ('invented_evidence',{'evidence_ids':['not-observed']},None,0.,'unproven_evidence'),
                        ('reward_mutation',{'reward':1},None,0.,'invalid_proposal'),
                        ('late_stop',{},'stopped',0.,'inactive_goal'),
                        ('late_completed',{},'completed',0.,'inactive_goal'),
                        ('expired_reply',{},None,31.,'request_expired'),
                    )
                    for name,patch,status,delay,expected in variants:
                        path = output/'cases'/(event_id+'-'+name+'.json')
                        write_new(path,history.snapshot())  # independent counterfactual branches
                        branch = ReplanGuard(PlanHistory(path),capabilities)
                        current = copy.deepcopy(snapshot)
                        if status is not None:
                            current['status'] = status
                        proposed = dict(base,**patch)
                        verdict = branch.review(request['request_id'],proposed,current,
                                                now_elapsed_s=snapshot['elapsed_s']+delay)
                        assert not verdict['accepted'] and not verdict['execution_authorized']
                        assert expected in verdict['reasons']
                        assert 'inputs_not_released' in verdict['reasons'] and 'unproven_generation' in verdict['reasons']
                        restored = PlanHistory(path)
                        assert restored.snapshot() == branch.history.snapshot()
                        result['cases'].append(dict(event_id=event_id,case=name,expected=expected,verdict=verdict,
                                                    proposal_source='synthetic fixture, not an LLM response'))
            assert PlanHistory(history.path).snapshot() == history.snapshot()
            for name,sha in spec['sha256'].items():
                assert file_sha256(run_dir/name) == sha
        assert len(result['events']) == 5 and len(result['cases']) == 35
        assert len(result['no_stall_reference_runs']) == 10
        for path,sha in frozen.items():
            assert file_sha256(root/path) == sha
        result['completed'] = True
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        write_new(output/'report.json',result)
    print(json.dumps(dict(completed=True,archived_stalls=5,synthetic_rejections=35,
                          no_stall_requests=0,llm_calls=0,game_actions=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
