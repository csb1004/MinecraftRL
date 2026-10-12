"""Actual GPU forward/state/logprob/gradient equivalence; no optimizer or game."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from minecraftrl.bounded_burnin import cache_length, forward_window_bounded
from minecraftrl.correction_bc import supervised_batch
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail
from minecraftrl.rl_policy import guided_minibatch, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import episode_rows, forward_window
from minecraftrl.world_store import digest


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/bounded-burnin-verification-20261009'
    out.mkdir(exist_ok=False)
    stop = root/'results/STOP'
    report = dict(completed=False, error=None, rows=[], optimizer_updates=0, game_actions=0)
    frozen = {}
    started = time.monotonic()

    def freeze(path, expected=None):
        value = digest(path)
        if expected is not None: assert value == expected, str(path)
        frozen[path.relative_to(root).as_posix()] = value

    def check_stop():
        if not stop.exists() or digest(stop) != report['stop_sha256']:
            raise InterruptedError('STOP changed during read-only equivalence probe')

    try:
        report['stop_sha256'] = digest(stop)
        assert stop.read_text().startswith(('RL training ended', 'RL collection ended'))
        torch.set_num_threads(1)
        torch.manual_seed(9610)
        torch.cuda.reset_peak_memory_stats()
        features = root/'results/masked-correction-features-20261009/features.pt'
        receipt = json.loads((features.parent/'features.json').read_text())
        assert receipt['completed']
        freeze(features, receipt['features_sha256'])
        data = torch.load(features, map_location='cpu')
        groups = episode_rows(data)
        for name in ('docs/bounded-burnin-protocol-2026-10-09.md', 'docs/research-spec.md',
                     'src/minecraftrl/bounded_burnin.py', 'src/minecraftrl/temporal_bc.py',
                     'scripts/verify_bounded_burnin.py', 'tests/test_bounded_burnin.py', 'weights/steve1/2x.model',
                     'external/steve1/steve1/VPT/lib/masked_attention.py',
                     'external/steve1/steve1/VPT/lib/util.py', 'external/steve1/steve1/VPT/lib/xf.py'):
            freeze(root/name)
        paths = dict(parent=root/'results/masked-correction-training-20261009/checkpoint.pt',
                     focused=root/'results/onset-bc-focused-20261009/checkpoint.pt')
        for path in paths.values():
            training = json.loads((path.parent/'training.json').read_text())
            assert training['completed']
            freeze(path, training['checkpoint_sha256'])
        sys.path.insert(0, str(root/'external/steve1'))
        from gym3.types import DictType
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
        kwargs, head_kwargs = model_kwargs(root/'weights/steve1/2x.model')
        skeleton = MinecraftAgentPolicy(space, kwargs, head_kwargs)
        blocks = skeleton.net.recurrent_layer.blocks
        net = SimpleNamespace(lastlayer=skeleton.net.lastlayer, final_ln=skeleton.net.final_ln,
                              recurrent_layer=SimpleNamespace(blocks=[None]*(len(blocks)-1)+[blocks[-1]]))
        del blocks, skeleton
        block = net.recurrent_layer.blocks[-1].cuda().eval()
        tail = torch.nn.Sequential(net.lastlayer, net.final_ln).cuda().eval()
        head = make_action_head(space, data['rows'][0]['features'].shape[-1], **head_kwargs).cuda().eval()
        params = list(block.parameters())+list(tail.parameters())+list(head.parameters())
        assert cache_length(block) == 128
        report['cache_length'] = 128
        print('verify actual GPU outputs, state and gradients', flush=True)
        for model, path in paths.items():
            saved = torch.load(path, map_location='cpu')
            load_tail(net, saved)
            head.load_state_dict(saved['actor'])
            del saved
            first_rows = next(iter(groups.values()))
            with torch.no_grad():
                forward_window(block, tail, first_rows, data['temporal_initial_memory'], 0, 8)
            for index, (name, original) in enumerate(groups.items()):
                onset = next(i for i, row in enumerate(original) if row['loss_eligible'])
                for kind, start, end in (('onset', onset-7, onset+1), ('late', len(original)-8, len(original))):
                    rows = list(original)
                    for i in range(start, end):
                        rows[i] = dict(rows[i], temporal_input=rows[i]['temporal_input'].cuda().detach().requires_grad_(True))
                    if index == 0:
                        # Warm both target-batch shapes including backward, with
                        # real labels and no optimizer, before recording timings.
                        warm = forward_window(block, tail, rows, data['temporal_initial_memory'], start, end)
                        selected, _, target = supervised_batch(rows[start:end], warm)
                        (-head.logprob(target, guided_minibatch(head, selected)).mean()).backward()
                        torch.cuda.synchronize()
                        del warm, selected, target
                    outputs, times = {}, {}
                    order = ('full', 'bounded') if (index+(kind == 'late')) % 2 == 0 else ('bounded', 'full')
                    for mode in order:
                        check_stop()
                        for p in params: p.grad = None
                        for i in range(start, end): rows[i]['temporal_input'].grad = None
                        memory = {}
                        calls = 0
                        expected_calls = end if mode == 'full' else end-max(0, start-128)
                        def capture(_module, _args, result):
                            nonlocal calls
                            calls += 1
                            if calls == expected_calls:
                                memory['state'] = result[1]
                        hook = block.register_forward_hook(capture)
                        torch.cuda.synchronize()
                        t = time.monotonic()
                        try:
                            method = forward_window if mode == 'full' else forward_window_bounded
                            f = method(block, tail, rows, data['temporal_initial_memory'], start, end)
                            selected, _, target = supervised_batch(rows[start:end], f)
                            pd = guided_minibatch(head, selected)
                            logp = head.logprob(target, pd)
                            (-logp.mean()).backward()
                            torch.cuda.synchronize()
                            times[mode] = time.monotonic()-t
                        finally:
                            hook.remove()
                        assert calls == expected_calls and 'state' in memory
                        outputs[mode] = dict(features=f.detach().cpu(), logp=logp.detach().cpu(),
                                             memory=state_fingerprint(memory['state']),
                                             parameter_gradients=[None if p.grad is None else p.grad.detach().cpu().clone() for p in params],
                                             input_gradients=[rows[i]['temporal_input'].grad.detach().cpu().clone() for i in range(start, end)])
                        del f, selected, target, pd, logp, memory
                    a, b = outputs['full'], outputs['bounded']
                    feature_error = float((a['features']-b['features']).abs().max())
                    logp_error = float((a['logp']-b['logp']).abs().max())
                    gradient_error = 0.
                    for ga, gb in zip(a['parameter_gradients']+a['input_gradients'], b['parameter_gradients']+b['input_gradients']):
                        assert (ga is None) == (gb is None)
                        if ga is not None:
                            assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
                            gradient_error = max(gradient_error, float((ga-gb).abs().max()))
                    assert feature_error == logp_error == gradient_error == 0 and a['memory'] == b['memory']
                    entry = dict(model=model, session=name, kind=kind, start=start, end=end,
                                 feature_max_error=feature_error, logprob_max_error=logp_error,
                                 gradient_max_error=gradient_error, memory_identical=True,
                                 full_seconds=times['full'], bounded_seconds=times['bounded'],
                                 speedup=times['full']/times['bounded'],
                                 full_burnin_frames=start, bounded_burnin_frames=min(start, 128))
                    report['rows'].append(entry)
                    write_json(out/'report.json', report)
                    print(json.dumps(entry), flush=True)
                    del outputs, a, b, rows
        for path, sha in frozen.items(): assert digest(root/path) == sha
        check_stop()
        assert len(report['rows']) == 24
        report.update(completed=True, frozen_sha256=frozen, sources_unchanged=True,
                      scope='Exact two-checkpoint/six-episode GPU comparison; no learning or rollout')
    except BaseException as exc:
        report['error'] = repr(exc)
        (out/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_s'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        write_json(out/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows', 'frozen_sha256')}), flush=True)
    return 0 if report['completed'] else 2


if __name__ == '__main__': raise SystemExit(main())
