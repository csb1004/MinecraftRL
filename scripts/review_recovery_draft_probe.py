"""Read-only archived LLM response review; no provider, scheduler, or game import."""
from copy import deepcopy
import json
from pathlib import Path

from minecraftrl.recovery_draft import validate_draft
from minecraftrl.replanning import PlanHistory, ReplanGuard
from minecraftrl.semantic import response_text
from minecraftrl.world_store import digest


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'results/recovery-draft-preparation-20261009'
    paid = base / 'paid-probe-transportfix'
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    manifest = read(base / 'manifest.json')
    report = read(paid / 'report.json')
    assert report['completed'] and report['network_requests'] == 2
    assert report['remaining_authorized_calls'] == 0 and report['new_game_actions'] == 0
    assert report['manifest_sha256'] == digest(base / 'manifest.json')
    output = base / 'planner-review'
    output.mkdir(exist_ok=False)
    sources = {str(p.relative_to(root)): digest(p) for p in (
        Path(__file__), base / 'manifest.json', paid / 'report.json', root / 'results/STOP',
        root / 'docs/research-spec.md', root / 'src/minecraftrl/replanning.py',
        root / 'src/minecraftrl/recovery_draft.py')}
    cases = []
    for case, attempt in zip(manifest['cases'], report['attempts']):
        label = case['label']
        assert attempt['label'] == label and attempt['status'] == 'contract_valid'
        input_path = Path(case['input'])
        response_path = paid / (label + '-response.json')
        history_path = root / 'results/goal-recovery-comparison-20261009/planner-evidence' / (label + '-history.json')
        for path in (input_path, response_path, history_path):
            sources[str(path.relative_to(root))] = digest(path)
        assert digest(input_path) == case['sha256']
        assert digest(response_path) == attempt['response_sha256']
        packet = read(input_path)
        raw_draft = json.loads(response_text(read(response_path)))
        draft = validate_draft(raw_draft, packet)
        assert draft == attempt['proposal'] and draft['choice'] == 'explore_then_resume'
        original_history = PlanHistory(history_path).snapshot()
        assert len(original_history['requests']) == 1
        request = original_history['requests'][0]
        snapshot = request['snapshot']
        assert packet['evidence_id'] == request['evidence']['event_id']
        assert packet['root_reference'] == snapshot['goal_ref']
        assert packet['assessment']['facts']['tick'] == snapshot['tick']
        assert abs(packet['remaining_control_s'] - (180 - snapshot['elapsed_s'])) < 1e-6
        # No runtime generation/release proof is invented for an archived alarm.
        assert snapshot['generation'] is None and snapshot['release_sequence'] is None
        assert snapshot['held_key_count'] is None
        proposal = dict(request_id=draft['request_id'], skill_id=draft['choice'], arguments={},
            budget_s=30., budget_ticks=draft['recovery_budget_ticks'],
            evidence_ids=[draft['evidence_id']], reason=draft['reason'][:240])
        reviews = []
        scenarios = (
            ('archived_id', 'wrong_request'),
            ('offline_id_binding_only', 'unsupported_skill'),
            ('stop_during_wait', 'inactive_goal'),
            ('user_pause_during_wait', 'inactive_goal'),
            ('inventory_satisfied_during_wait', 'goal_already_satisfied'),
            ('different_world', 'stale_context'),
            ('changed_target', 'changed_preconditions'),
            ('expired_reply', 'request_expired'),
            ('session_budget_exhausted', 'budget_exceeds_remaining_time'),
            ('invented_evidence', 'unproven_evidence'),
            ('injected_reward', 'invalid_proposal'),
        )
        for name, expected_reason in scenarios:
            history = PlanHistory()  # Independent in-memory counterfactual, never saved over originals.
            history.data = deepcopy(original_history)
            guard = ReplanGuard(history, request['capabilities'])
            current = deepcopy(snapshot)
            value = deepcopy(proposal)
            if name != 'archived_id':
                # Explicit analysis mapping only. This does not bind a real live request.
                value['request_id'] = request['request_id']
            now = snapshot['elapsed_s'] + attempt['elapsed_s']
            if name == 'stop_during_wait': current['status'] = 'stopped'
            if name == 'user_pause_during_wait': current['status'] = 'user_paused'
            if name == 'inventory_satisfied_during_wait': current['facts']['goal_inventory'] += 1
            if name == 'different_world': current['world_id'] = 'counterfactual-different-world'
            if name == 'changed_target': current['facts']['target'] = [0, 64, 0]
            if name == 'expired_reply': now = snapshot['elapsed_s'] + request['ttl_s']
            if name == 'session_budget_exhausted': now = 180.
            if name == 'invented_evidence': value['evidence_ids'] = ['counterfactual-invented']
            if name == 'injected_reward': value['reward'] = 1
            verdict = guard.review(request['request_id'], value, current, now_elapsed_s=now)
            assert expected_reason in verdict['reasons'], (label, name, verdict)
            assert {'unproven_generation', 'inputs_not_released'} <= set(verdict['reasons'])
            assert not verdict['accepted'] and not verdict['execution_authorized']
            assert verdict['rm_effect'] == 'preserve' and verdict['goal_ref'] == snapshot['goal_ref']
            again = guard.review(request['request_id'], value, current, now_elapsed_s=now)
            assert again['reasons'] == ['request_consumed']
            assert history.data['failures'] == original_history['failures']
            assert history.data['requests'][0]['snapshot'] == snapshot
            reviews.append(dict(scenario=name, expected_reason=expected_reason, verdict=verdict,
                counterfactual=name != 'archived_id', replay_rejected=True))
        cases.append(dict(label=label, choice=draft['choice'], api_latency_s=attempt['elapsed_s'],
            remaining_after_observed_latency_s=packet['remaining_control_s']-attempt['elapsed_s'],
            fixed_explore_rule_agreement=True, reviews=reviews,
            original_llm_request_id=draft['request_id'], original_guard_request_id=request['request_id'],
            review_budget_s=30., review_budget_origin='Local diagnostic value, not an LLM or execution budget',
            reason_projection='First 240 characters for existing guard; full original reply preserved'))
    for relative, sha in sources.items():
        assert digest(root / relative) == sha, relative
    totals = {key: sum(a['usage'][key] for a in report['attempts'])
              for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    result = dict(verified=True, actual_api_calls=2, additional_api_calls=0, api_retries=0,
        actual_games=0, optimizer_updates=0, new_game_actions=0,
        local_review_cases=sum(len(c['reviews']) for c in cases), usage=totals,
        cases=cases, source_sha256=sources, execution_authorized=False,
        scope='Two real archived API replies plus explicit offline counterfactual guard checks; no live handoff',
        next_runtime_gaps=['input release and pause acknowledgement before live request',
            'execution generation binding and current observation recheck',
            'opt-in diagnostic capability binding, one-window budget, and original-goal return',
            'nonblocking STOP during actual API wait; separate paid scope required'])
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(verified=True, real_api_replies=2, offline_reviews=result['local_review_cases'],
                         additional_api_calls=0, new_game_actions=0, usage=totals)))


if __name__ == '__main__':
    main()
