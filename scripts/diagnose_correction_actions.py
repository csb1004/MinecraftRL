"""One-use recurrent BC/command diagnosis; no optimizer, environment, or STOP rearm."""
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from minecraftrl.correction_diagnostics import CommandMetrics, aggregate, annotate_correction
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail
from minecraftrl.rl_policy import guided, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import device_tree, episode_rows, latent_features, recurrent_step
from minecraftrl.world_store import digest


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/correction-action-diagnosis-20261009'
    out.mkdir(exist_ok=False)
    stop = root / 'results/STOP'
    report = dict(completed=False, error=None, optimizer_updates=0, game_actions=0, new_rollouts=0,
                  scope='Recorded trajectory predictions, not closed-loop policy success')
    started = time.monotonic()
    frozen, buckets = {}, defaultdict(list)

    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def freeze(path, expected=None):
        path = Path(path)
        sha = digest(path)
        if expected is not None:
            assert sha == expected, str(path)
        frozen[path.relative_to(root).as_posix()] = sha
        return sha

    def check_stop():
        if not stop.exists() or digest(stop) != report['stop_sha256']:
            raise InterruptedError('STOP changed during read-only diagnosis')

    def add(model, split, session, kind, group, values):
        buckets[(model, split, session, kind, group)].append(values)

    try:
        report['stop_sha256'] = digest(stop)
        assert stop.read_text().startswith(('RL training ended', 'RL collection ended'))
        torch.set_num_threads(1)
        torch.manual_seed(9604)
        torch.cuda.reset_peak_memory_stats()
        for name in ('docs/correction-action-diagnosis-protocol-2026-10-09.md',
                     'docs/correction-action-diagnosis-clarification-2026-10-09.md', 'docs/research-spec.md',
                     'src/minecraftrl/correction_diagnostics.py', 'scripts/diagnose_correction_actions.py',
                     'tests/test_correction_diagnostics.py', 'weights/steve1/2x.model'):
            freeze(root / name)
        features = root / 'results/masked-correction-features-20261009/features.pt'
        training_path = root / 'results/masked-correction-training-20261009/training.json'
        training = read(training_path)
        freeze(training_path)
        freeze(features, training['features_sha256'])
        data = torch.load(features, map_location='cpu')
        groups = episode_rows(data)
        freeze(root / 'configs/masked-correction-bc-20261009.json', data['plan_sha256'])
        freeze(root / 'weights/steve1/steve1.weights', data['model_sha256'])
        paths = dict(parent=root / data['plan']['parent'], candidate=training_path.parent / 'checkpoint.pt')
        freeze(paths['parent'], training['parent_sha256'])
        freeze(paths['candidate'], training['checkpoint_sha256'])
        sessions = {s['id']: s for s in data['plan']['sessions']}
        annotations = {}
        for name, session in sessions.items():
            run = root / session['run']
            for filename, sha in session['files_sha256'].items():
                freeze(run / filename, sha)
            source = read(run / 'correction-rows.json')
            world = read(run / 'summary.json')['initial_world']
            annotations[name] = annotate_correction(source, world, session['handoffs'][0]['waypoints'])
            assert len(annotations[name]) == len(groups[name])
        del source
        write_json(out / 'annotations.json', annotations)
        print('source routes verified; initialize frozen inference modules', flush=True)
        write_json(out / 'report.json', report)

        sys.path.insert(0, str(root / 'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.actions import Buttons
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        mapper = CameraHierarchicalMapping(n_camera_bins=11)
        space = DictType(**mapper.get_action_space_update())
        kwargs, head_kwargs = model_kwargs(root / 'weights/steve1/2x.model')
        skeleton = MinecraftAgentPolicy(space, kwargs, head_kwargs)
        blocks = skeleton.net.recurrent_layer.blocks
        net = SimpleNamespace(lastlayer=skeleton.net.lastlayer, final_ln=skeleton.net.final_ln,
                              recurrent_layer=SimpleNamespace(blocks=[None]*(len(blocks)-1)+[blocks[-1]]))
        del blocks, skeleton
        block = net.recurrent_layer.blocks[-1].cuda().eval().requires_grad_(False)
        tail = torch.nn.Sequential(net.lastlayer, net.final_ln).cuda().eval().requires_grad_(False)
        head = make_action_head(space, data['rows'][0]['features'].shape[-1], **head_kwargs).cuda().eval().requires_grad_(False)
        metrics = CommandMetrics(mapper.BUTTON_IDX_TO_FACTORED, mapper.BUTTON_IDX_TO_CAMERA_META_OFF,
                                 mapper.CAMERA_IDX_TO_FACTORED, Buttons.ALL, mapper.camera_null_bin, 'cuda')
        report.update(button_categories=len(mapper.BUTTON_IDX_TO_FACTORED), camera_categories=len(mapper.CAMERA_IDX_TO_FACTORED))
        max_replay_error = 0.
        with torch.no_grad(), (out / 'demonstration-predictions.jsonl').open('x') as log:
            for model, path in paths.items():
                saved = torch.load(path, map_location='cpu')
                load_tail(net, saved)
                head.load_state_dict(saved['actor'])
                del saved
                for name, rows in groups.items():
                    check_stop()
                    memory = device_tree(data['temporal_initial_memory'], 'cuda')
                    split = sessions[name]['split']
                    for i, (row, label) in enumerate(zip(rows, annotations[name])):
                        check_stop()
                        latent, memory = recurrent_step(block, row, memory, i)
                        f = latent_features(tail, latent, row)
                        if model == 'parent':
                            max_replay_error = max(max_replay_error, float((f-row['features'].cuda()).abs().max()))
                        values = metrics.measure(guided(head, f), row['action'])
                        add(model, split, 'all', 'owner', label['owner'], values)
                        if label['owner'] == 'teacher':
                            for session_key in ('all', name):
                                add(model, split, session_key, 'leg', label['leg'], values)
                                add(model, split, session_key, 'leg_motion', label['leg']+'/'+label['motion'], values)
                                if label['teacher_index'] < 8:
                                    add(model, split, session_key, 'onset', 'first8', values)
                                if label['teacher_index'] < 32:
                                    add(model, split, session_key, 'onset', 'first32', values)
                        elif label['pre_handoff_last32']:
                            add(model, split, 'all', 'context_window', 'last32_student', values)
                        log.write(json.dumps(dict(model=model, session=name, split=split, frame=i, label=label, metrics=values))+'\n')
                    print(json.dumps(dict(model=model, session=name, frames=len(rows))), flush=True)
        assert max_replay_error < 1e-5
        report['parent_feature_replay_max_error'] = max_replay_error
        report['nll_crosscheck'] = []
        for model, prior in (('parent', 'before'), ('candidate', 'after')):
            for split in ('train', 'development'):
                stats = aggregate(buckets[(model, split, 'all', 'owner', 'teacher')])
                expected = training[prior][split]['nll']
                assert stats['nll']['count'] == training['label_counts'][split]
                error = abs(stats['nll']['mean'] - expected)
                assert error < 1e-5, (model, split, error)
                report['nll_crosscheck'].append(dict(model=model, split=split, nll=stats['nll']['mean'], error=error))

        print('same-trajectory diagnosis complete; inspect existing live-policy probabilities', flush=True)
        comparison = root / 'results/masked-correction-comparison-20261009'
        batch = read(comparison / 'experiment.json')
        audit = read(comparison / 'audit.json')
        assert batch['completed'] and audit['verified']
        freeze(comparison / 'experiment.json')
        freeze(comparison / 'audit.json')
        live_reports = []
        with torch.no_grad(), (out / 'live-predictions.jsonl').open('x') as log:
            for trial in batch['runs']:
                if trial['world']['kind'] != 'long_wall':
                    continue
                model = trial['label'].split('-')[-1]
                # Source records use Windows paths; this analysis runs in WSL.
                run = root / trial['path'].replace('\\','/').split('/MinecraftRL/',1)[-1]
                prior_audit = next(a for a in audit['rows'] if a['label'] == trial['label'])
                for filename, sha in prior_audit['source_sha256'].items():
                    freeze(run / filename, sha)
                trajectory = torch.load(run / 'trajectory.pt', map_location='cpu')
                saved = torch.load(paths[model], map_location='cpu')
                for key in ('actor', 'value', 'backbone_tail', 'backbone_temporal'):
                    assert state_fingerprint(saved[key]) == state_fingerprint(trajectory['initial_heads'][key])
                head.load_state_dict(saved['actor'])
                del saved
                initial = read(run / 'initial-state.json')
                previous = initial['server']
                oz = initial['initial_world']['chunk'][1]*16
                assert not initial['initial_world']['curriculum']['mirror_z']
                transitions = [json.loads(line) for line in (run / 'events.jsonl').open()]
                transitions = [e for e in transitions if e['event'] == 'transition']
                assert len(transitions) == len(trajectory['rows']) == trial['report']['steps']
                entered = False
                error = 0.
                for i, (row, event) in enumerate(zip(trajectory['rows'], transitions)):
                    check_stop()
                    pd = guided(head, row['features'].cuda())
                    action = {k: v.cuda() for k,v in row['action'].items()}
                    error = max(error, abs(float(head.logprob(action,pd).item()) - row['old_logp']))
                    values = metrics.measure(pd)
                    pressed = lambda k: bool(event['action'][k][0])
                    values.update(actual_move=int(any(pressed(k) for k in ('forward','back','left','right'))),
                                  actual_attack=int(pressed('attack')))
                    group = 'closed_corner' if previous['server_z']-oz <= 1.35 else 'after_recovery' if entered else 'before_recovery'
                    add(model, 'live_eval', trial['label'], 'live_region', group, values)
                    log.write(json.dumps(dict(model=model, frame=i, group=group, metrics=values))+'\n')
                    entered |= event['info']['rl_reward']['recovery']['event'] == 'entered'
                    previous = event['info']['workshop']
                assert error < 1e-5, (model, error)
                live_reports.append(dict(model=model, frames=len(transitions), max_own_logprob_error=error))
                del trajectory, transitions
                print(json.dumps(live_reports[-1]), flush=True)
        report['live_replay'] = live_reports
        summaries = [dict(model=k[0], split=k[1], session=k[2], kind=k[3], group=k[4], frames=len(v), metrics=aggregate(v))
                     for k,v in sorted(buckets.items())]
        write_json(out / 'metrics.json', summaries)
        check_stop()
        for name, sha in frozen.items():
            assert digest(root / name) == sha, name
        report.update(completed=True, frozen_sha256=frozen, sources_unchanged=True,
                      demonstration_frames_per_model=sum(len(r) for r in groups.values()),
                      results_sha256={name:digest(out/name) for name in
                                      ('annotations.json','metrics.json','demonstration-predictions.jsonl','live-predictions.jsonl')})
    except BaseException as exc:
        report['error'] = repr(exc)
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_s'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        write_json(out / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('frozen_sha256','results_sha256')}))
    return 0 if report['completed'] else 2


if __name__ == '__main__':raise SystemExit(main())
