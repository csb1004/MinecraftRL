"""Coverage-expanded onset BC; exact finite-cache burn-in, unchanged development data."""
import argparse
import copy
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from minecraftrl.bc_core import comparison_gate
from minecraftrl.bounded_burnin import forward_window_bounded
from minecraftrl.correction_bc import supervised_batch
from minecraftrl.correction_diagnostics import aggregate
from minecraftrl.onset_bc import OnsetMetrics, correction_schedule
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import TEMPORAL_SCOPE, load_tail, tail_state, temporal_state
from minecraftrl.rl_control import claim_automatic_stop, publish_completion
from minecraftrl.rl_policy import cpu_tree, guided, guided_minibatch, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import device_tree, episode_rows, latent_features, recurrent_step
from minecraftrl.world_store import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if not out.is_relative_to(root/'results'):
        raise ValueError('project-local result required')
    out.mkdir(exist_ok=False)
    stop = root/'results/STOP'
    claim_automatic_stop(stop, out/'STOP-before.txt')
    started = time.monotonic()
    report = dict(completed=False, error=None, mode='coverage_onset', updates=0, game_actions=0,
                  learning_rate=5e-6, window_size=8, order_seed=9612,
                  diagnostic_min_development_onset_turn_probability=.005,
                  runtime_comparison_eligible=False, diagnostic_eligible=False)
    frozen = {}

    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def freeze(path, expected=None):
        sha = digest(path)
        if expected is not None:
            assert sha == expected, str(path)
        frozen[path.relative_to(root).as_posix()] = sha
        return sha

    def check_stop():
        if stop.exists():
            raise InterruptedError('STOP during onset BC')

    try:
        torch.set_num_threads(1)
        torch.manual_seed(9612)
        torch.cuda.reset_peak_memory_stats()
        for name in ('docs/onset-coverage-training-protocol-2026-10-09.md', 'docs/research-spec.md',
                     'src/minecraftrl/onset_bc.py', 'src/minecraftrl/temporal_bc.py',
                     'src/minecraftrl/bounded_burnin.py', 'tests/test_bounded_burnin.py',
                     'src/minecraftrl/correction_bc.py', 'src/minecraftrl/correction_diagnostics.py',
                     'scripts/train_coverage_bc.py', 'scripts/audit_coverage_bc_training.py', 'tests/test_onset_bc.py'):
            freeze(root/name)
        proof = root/'results/bounded-burnin-verification-20261009/report.json'
        verified = read(proof)
        assert verified['completed'] and verified['error'] is None and len(verified['rows']) == 24
        assert all(r['gradient_max_error'] == r['feature_max_error'] == r['logprob_max_error'] == 0 and r['memory_identical'] for r in verified['rows'])
        freeze(proof)
        freeze(root/'src/minecraftrl/bounded_burnin.py', verified['frozen_sha256']['src/minecraftrl/bounded_burnin.py'])
        features = root/'results/onset-coverage-features-20261009/features.pt'
        receipt = read(features.parent/'features.json')
        source_audit = read(features.parent/'source-replay-audit.json')
        assert receipt['completed'] and source_audit['verified'] and source_audit['features_sha256'] == receipt['features_sha256']
        freeze(features.parent/'source-replay-audit.json')
        freeze(features, receipt['features_sha256'])
        data = torch.load(features, map_location='cpu')
        groups = episode_rows(data)
        freeze(root/'configs/onset-coverage-bc-20261009.json', data['plan_sha256'])
        old_plan_path = root/'configs/masked-correction-bc-20261009.json'
        freeze(old_plan_path, '347d327f5d3673c31144e0c4ee9e1f2f053f6b9d58d61882fd6a22d9b62ee99b')
        old_plan = read(old_plan_path)
        assert data['plan']['sessions'][:len(old_plan['sessions'])] == old_plan['sessions']
        added = data['plan']['sessions'][len(old_plan['sessions']):]
        assert added and all(s['split'] == 'train' for s in added)
        freeze(root/'weights/steve1/steve1.weights', data['model_sha256'])
        freeze(root/'weights/steve1/2x.model')
        parent_path = root/'results/masked-correction-training-20261009/checkpoint.pt'
        prior = read(parent_path.parent/'training.json')
        assert prior['completed']
        freeze(parent_path, 'ed39af3d0df57ebcc6c8b1b98f7aa12d836d5a7cf120d1fce1911f884da29525')
        freeze(parent_path.parent/'training.json')
        parent = torch.load(parent_path, map_location='cpu')
        assert parent['scope'] == TEMPORAL_SCOPE and parent['base_model_sha256'] == data['model_sha256']
        assert parent['condition']['reward_mode'] == 'wood-potential-v4'
        sessions = {s['id']: s for s in data['plan']['sessions']}
        for session in sessions.values():
            for name, sha in session['files_sha256'].items():
                freeze(root/session['run']/name, sha)
        splits = {name: s['split'] for name, s in sessions.items()}
        onset = {name: next(i for i, r in enumerate(rows) if r['loss_eligible']) for name, rows in groups.items()}
        schedule = correction_schedule(groups, splits, 'onset', seed=9612)
        assert 260 < len(schedule) <= 2400, 'bounded one-pass-plus-onset budget exceeded'
        milestones = {math.ceil(len(schedule)*i/5) for i in range(1, 6)}
        write_json(out/'schedule.json', schedule)

        sys.path.insert(0, str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.actions import Buttons
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        mapper = CameraHierarchicalMapping(n_camera_bins=11)
        space = DictType(**mapper.get_action_space_update())
        kwargs, head_kwargs = model_kwargs(root/'weights/steve1/2x.model')
        skeleton = MinecraftAgentPolicy(space, kwargs, head_kwargs)
        load_tail(skeleton.net, parent)
        blocks = skeleton.net.recurrent_layer.blocks
        net = SimpleNamespace(lastlayer=skeleton.net.lastlayer, final_ln=skeleton.net.final_ln,
                              recurrent_layer=SimpleNamespace(blocks=[None]*(len(blocks)-1)+[blocks[-1]]))
        del blocks, skeleton
        block = net.recurrent_layer.blocks[-1].cuda().eval().requires_grad_(True)
        tail = torch.nn.Sequential(net.lastlayer, net.final_ln).cuda().eval().requires_grad_(True)
        actor = make_action_head(space, data['rows'][0]['features'].shape[-1], **head_kwargs).cuda().eval()
        actor.load_state_dict(parent['actor'])
        reference = copy.deepcopy(actor).requires_grad_(False)
        params = list(block.parameters())+list(tail.parameters())+list(actor.parameters())
        optimizer = torch.optim.AdamW(params, lr=5e-6, weight_decay=0, foreach=False)
        metrics = OnsetMetrics(mapper.BUTTON_IDX_TO_FACTORED, mapper.BUTTON_IDX_TO_CAMERA_META_OFF,
                               mapper.CAMERA_IDX_TO_FACTORED, Buttons.ALL, mapper.camera_null_bin, 'cuda')
        initial = data['temporal_initial_memory']
        reference_features = {name: [] for name in groups}
        report.update(features_sha256=digest(features), parent_sha256=digest(parent_path),
                      model_sha256=data['model_sha256'], plan_sha256=data['plan_sha256'],
                      planned_updates=len(schedule), schedule_sha256=digest(out/'schedule.json'),
                      label_counts={split: sum(r['loss_eligible'] for name, rows in groups.items()
                                              if splits[name] == split for r in rows)
                                    for split in ('train', 'development')},
                      trainable_parameters=sum(p.numel() for p in params))

        def measure(*, create_reference=False, first_only=False, capture=False):
            buckets = defaultdict(list)
            first = {}
            captured = []
            with torch.no_grad():
                for name, rows in groups.items():
                    if first_only and splits[name] != 'train':
                        continue
                    memory = device_tree(initial, 'cuda')
                    for i, row in enumerate(rows):
                        check_stop()
                        latent, memory = recurrent_step(block, row, memory, i)
                        f = latent_features(tail, latent, row)
                        if create_reference:
                            reference_features[name].append(cpu_tree(f))
                        if first_only and i != onset[name]:
                            continue
                        pd = guided(actor, f)
                        assert torch.isfinite(f).all() and all(torch.isfinite(v).all() for v in pd.values())
                        values = metrics.measure(pd, row['action'])
                        ref = guided(reference, reference_features[name][i].cuda())
                        values['reference_kl'] = float(actor.kl_divergence(ref, pd).item())
                        split = splits[name]
                        buckets[(split, 'all')].append(values)
                        owner = 'teacher' if row['loss_eligible'] else 'context'
                        buckets[(split, owner)].append(values)
                        if row['loss_eligible']:
                            buckets[(split, 'first' if i == onset[name] else 'continuation')].append(values)
                            if i < onset[name]+8:
                                buckets[(split, 'first8')].append(values)
                        if i == onset[name]:
                            first[name] = values
                        if capture and split == 'development':
                            captured.append(cpu_tree(f))
                        if first_only:
                            break
            summaries = {split: {group: aggregate(values) for (s, group), values in buckets.items() if s == split}
                         for split in ('train', 'development') if any(s == split for s, _ in buckets)}
            return dict(splits=summaries, first_actions=first), captured

        print('measure current parent; cache its own reference features', flush=True)
        before, _ = measure(create_reference=True)
        for split in ('train', 'development'):
            if split == 'development':
                assert abs(before['splits'][split]['teacher']['nll']['mean']-prior['after'][split]['nll']) < 1e-5
            assert before['splits'][split]['teacher']['reference_kl']['mean'] == 0
        report['before'] = before
        write_json(out/'training.json', report)
        exposures = Counter()
        for unit in schedule:
            check_stop()
            name, start, end = unit['session'], unit['start'], unit['end']
            rows = groups[name]
            assert splits[name] == 'train'
            optimizer.zero_grad(set_to_none=True)
            f = forward_window_bounded(block, tail, rows, initial, start, end, check_stop=check_stop)
            f, _, action = supervised_batch(rows[start:end], f)
            indices = [i for i in range(start, end) if rows[i]['loss_eligible']]
            base = torch.cat([reference_features[name][i] for i in indices], 1).cuda()
            pd = guided_minibatch(actor, f)
            with torch.no_grad():
                ref = guided_minibatch(reference, base)
            nll = -actor.logprob(action, pd).mean()
            kl = actor.kl_divergence(ref, pd).mean()
            loss = nll+kl
            if not torch.isfinite(loss):
                raise ValueError('nonfinite onset BC loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            check_stop()
            optimizer.step()
            report['updates'] += 1
            for i in indices:
                assert rows[i]['owner'] == 'teacher'
                exposures[(name, i)] += 1
            entry = dict(unit, update=report['updates'], nll=float(nll), reference_kl=float(kl),
                         gradient_norm=float(norm), supervised_frames=len(indices),
                         context_frames=end-start-len(indices), elapsed_s=time.monotonic()-started)
            with (out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry)+'\n')
            del f, base, action, pd, ref, nll, kl, loss
            if report['updates'] % 10 == 0:
                print(json.dumps(entry), flush=True)
            if report['updates'] in milestones:
                diagnostic, _ = measure(first_only=True)
                write_json(out/f"train-onset-{report['updates']:03d}.json", diagnostic)
                print(json.dumps(dict(update=report['updates'], train_first=diagnostic['splits']['train']['first'])), flush=True)
                write_json(out/'training.json', report)
        optimizer.zero_grad(set_to_none=True)
        print('measure final full sequences', flush=True)
        after, captured = measure(capture=True)
        actor_saved, tail_saved, temporal_saved = cpu_tree(actor.state_dict()), tail_state(net), temporal_state(net)
        for key, saved in (('actor', actor_saved), ('backbone_tail', tail_saved), ('backbone_temporal', temporal_saved)):
            assert state_fingerprint(saved) != state_fingerprint(parent[key]), key
        candidate = dict(schema=1, scope=TEMPORAL_SCOPE, actor=actor_saved, value=parent['value'],
                         condition=parent['condition'], backbone_tail=tail_saved, backbone_temporal=temporal_saved,
                         base_model_sha256=data['model_sha256'], updates=parent.get('updates', 0)+report['updates'],
                         bc=dict(method='coverage_onset_bc', updates=report['updates'], window=8,
                                 burnin='exact_last_block_cache_current_weights', value_frozen=True, ppo_optimizer_resumable=False,
                                 parent_sha256=digest(parent_path), features_sha256=digest(features)))
        torch.save(candidate, out/'checkpoint.pt')
        torch.save(cpu_tree(optimizer.state_dict()), out/'bc-optimizer.pt')
        load_tail(net, parent)
        actor.load_state_dict(parent['actor'])
        restored = torch.load(out/'checkpoint.pt', map_location='cpu')
        load_tail(net, restored)
        actor.load_state_dict(restored['actor'])
        reloaded, second = measure(capture=True)
        reload_error = max(float((a-b).abs().max()) for a, b in zip(captured, second))
        assert len(captured) == len(second) == 1701 and reload_error == 0 and after == reloaded
        assert state_fingerprint(restored['value']) == state_fingerprint(parent['value'])
        assert restored['condition'] == parent['condition']
        a, b = before['splits']['development'], after['splits']['development']
        min_turn = min(v['p_release_attack_and_turn'] for name, v in after['first_actions'].items()
                       if splits[name] == 'development')
        eligible = (b['first']['nll']['mean'] <= a['first']['nll']['mean']-1 and
                    b['first']['p_release_attack_and_turn']['mean'] > a['first']['p_release_attack_and_turn']['mean'] and
                    b['continuation']['nll']['mean'] <= 1.1*a['continuation']['nll']['mean'] and
                    min_turn >= report['diagnostic_min_development_onset_turn_probability'])
        strict = comparison_gate(a['teacher']['nll']['mean'], b['teacher']['nll']['mean'],
                                 after['splits']['train']['teacher']['reference_kl']['mean'], 1.)
        assert len(exposures) == report['label_counts']['train']
        exposure_rows = [dict(session=k[0], frame=k[1], count=v) for k, v in sorted(exposures.items())]
        write_json(out/'target-exposures.json', exposure_rows)
        for name, sha in frozen.items():
            assert digest(root/name) == sha, name
        check_stop()
        report.update(completed=True, after=after, reload_max_error=reload_error, value_unchanged=True,
                      runtime_comparison_eligible=strict, diagnostic_eligible=eligible,
                      development_onset_min_turn_probability=min_turn,
                      diagnostic_gate_requires_full_runtime_replay=True,
                      total_teacher_target_exposures=sum(exposures.values()), unique_teacher_targets=len(exposures),
                      onset_exposures={name: exposures[(name, onset[name])] for name in groups if splits[name] == 'train'},
                      checkpoint_sha256=digest(out/'checkpoint.pt'), optimizer_sha256=digest(out/'bc-optimizer.pt'),
                      scope=TEMPORAL_SCOPE, frozen_sha256=frozen, sources_unchanged=True,
                      training_onset_states=sum(split == 'train' for split in splits.values()),
                      burnin='exact_last_block_cache_current_weights',
                      limitation='Expanded training data and update count; not an isolated sampling ablation or final evaluation.')
    except BaseException as exc:
        report['error'] = repr(exc)
        (out/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_s'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        write_json(out/'training.json', report)
        publish_completion(stop, 'RL training ended; coverage onset BC finished; inspect report.')
    print(json.dumps({k: v for k, v in report.items() if k not in ('before', 'after', 'frozen_sha256')}), flush=True)
    return 0 if report['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
