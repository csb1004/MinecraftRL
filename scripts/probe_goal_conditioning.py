"""One-use, observation-controlled goal intervention; no training or game actions."""
import argparse
import copy
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from minecraftrl.correction_diagnostics import aggregate
from minecraftrl.onset_bc import OnsetMetrics
from minecraftrl.policies import model_kwargs
from minecraftrl.policy_tail import load_tail
from minecraftrl.rl_control import claim_automatic_stop, publish_completion
from minecraftrl.rl_policy import RLPolicy, cpu_tree, guided, state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import device_tree, episode_rows
from minecraftrl.world_store import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
    training_dir = root/'results/onset-coverage-training-20261009'
    training = read(training_dir/'training.json')
    assert training['completed'] and training['error'] is None
    assert not training['diagnostic_eligible'], 'planned runtime comparison takes precedence'
    assert read(training_dir/'supervision-audit.json')['verified']
    out = args.output.resolve()
    if not out.is_relative_to(root/'results'):
        raise ValueError('project-local output required')
    out.mkdir(exist_ok=False)
    stop = root/'results/STOP'
    claim_automatic_stop(stop, out/'STOP-before.txt')
    started = time.monotonic()
    report = dict(completed=False, error=None, game_actions=0, optimizer_updates=0,
                  actual_llm_calls=0, rows=[], seeds=[9711, 9712, 9713],
                  prompts=dict(wood='collect wood', explore='go explore'),
                  window=8, cond_scale=6., frozen_sha256={})

    def freeze(path, expected=None):
        value = digest(path)
        if expected is not None:
            assert value == expected, str(path)
        report['frozen_sha256'][path.relative_to(root).as_posix()] = value
        return value

    def check_stop():
        if stop.exists():
            raise InterruptedError('STOP during goal conditioning diagnosis')

    try:
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats()
        for name in ('docs/goal-conditioning-diagnosis-protocol-2026-10-09.md',
                     'docs/research-spec.md', 'scripts/probe_goal_conditioning.py',
                     'src/minecraftrl/rl_policy.py', 'src/minecraftrl/onset_bc.py',
                     'src/minecraftrl/correction_diagnostics.py',
                     'external/steve1/steve1/run_agent/paper_prompts.py',
                     'external/steve1/steve1/utils/embed_utils.py',
                     'external/steve1/steve1/data/text_alignment/vae.py',
                     'weights/steve1/attn.pth', 'weights/steve1/steve1_prior.pt',
                     'weights/steve1/2x.model'):
            freeze(root/name)
        checkpoint_path = training_dir/'checkpoint.pt'
        freeze(checkpoint_path, training['checkpoint_sha256'])
        freeze(training_dir/'training.json')
        freeze(training_dir/'supervision-audit.json')
        features_path = root/'results/onset-coverage-features-20261009/features.pt'
        freeze(features_path, training['features_sha256'])
        data = torch.load(features_path, map_location='cpu')
        groups = episode_rows(data)
        assert len(groups) == 13
        freeze(root/'configs/onset-coverage-bc-20261009.json', data['plan_sha256'])
        freeze(root/'weights/steve1/steve1.weights', data['model_sha256'])
        original_path = root/data['plan']['embedding']
        freeze(original_path, data['goal_embedding_sha256'])
        embeddings = dict(original_wood=np.load(original_path, allow_pickle=False))
        np.save(out/'original-wood.npy', embeddings['original_wood'])
        sys.path.insert(0, str(root/'external/steve1'))
        from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
        from steve1.data.text_alignment.vae import load_vae_model
        from steve1.mineclip_code.load_mineclip import load
        from steve1.utils.embed_utils import get_prior_embed
        config = copy.deepcopy(MINECLIP_CONFIG)
        config['ckpt']['path'] = str(root/'weights/steve1/attn.pth')
        mineclip = load(config, device='cuda').eval()
        prior = load_vae_model(dict(PRIOR_INFO, model_path=str(root/'weights/steve1/steve1_prior.pt')))
        text_features = {}
        with torch.no_grad(), torch.cuda.amp.autocast():
            for name, prompt in report['prompts'].items():
                text_features[name] = mineclip.encode_text(prompt).detach().float()
                np.save(out/(name+'-text.npy'), text_features[name].cpu().numpy())
        manifest = []
        for seed in report['seeds']:
            check_stop()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            latent = torch.randn(1, prior.latent_dim, device='cuda')
            np.save(out/f'latent-{seed}.npy', latent.cpu().numpy())
            latent_hash = state_fingerprint(latent)
            for name, prompt in report['prompts'].items():
                with torch.no_grad(), torch.cuda.amp.autocast():
                    encoded = prior.decode(latent, text_features[name]).cpu().numpy()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                official = get_prior_embed(prompt, mineclip, prior, 'cuda')
                error = float(np.max(np.abs(encoded.astype(np.float32)-official.astype(np.float32))))
                assert encoded.shape == embeddings['original_wood'].shape and np.isfinite(encoded).all()
                assert error == 0 and state_fingerprint(latent) == latent_hash
                key = f'{name}_{seed}'
                embeddings[key] = encoded
                np.save(out/(key+'.npy'), encoded)
                manifest.append(dict(condition=key, seed=seed, prompt=prompt, latent_sha256=latent_hash,
                                     embedding_file_sha256=digest(out/(key+'.npy')), official_max_error=error))
        write_json(out/'embeddings.json', manifest)
        del mineclip, prior, text_features, latent, encoded, official
        gc.collect()
        torch.cuda.empty_cache()
        print('three paired latent draws verified against official prior', flush=True)

        from gym3.types import DictType
        from steve1.VPT.lib.action_head import make_action_head
        from steve1.VPT.lib.action_mapping import CameraHierarchicalMapping
        from steve1.VPT.lib.actions import Buttons
        from steve1.VPT.lib.torch_util import set_default_torch_device
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent
        from steve1.embed_conditioned_policy import MinecraftAgentPolicy
        set_default_torch_device('cpu')
        mapper = CameraHierarchicalMapping(n_camera_bins=11)
        space = DictType(**mapper.get_action_space_update())
        kwargs, head_kwargs = model_kwargs(root/'weights/steve1/2x.model')
        policy = MinecraftAgentPolicy(space, kwargs, head_kwargs)
        base = torch.load(root/'weights/steve1/steve1.weights', map_location='cpu')
        policy.load_state_dict(base, strict=True)
        del base
        saved = torch.load(checkpoint_path, map_location='cpu')
        load_tail(policy.net, saved)
        policy = policy.cuda().eval().requires_grad_(False)
        actor = make_action_head(space, data['rows'][0]['features'].shape[-1], **head_kwargs).cuda().eval()
        actor.load_state_dict(saved['actor'])
        actor.requires_grad_(False)
        from minecraftrl.cuda_startup import prime_convolution
        prime_convolution(policy.net.img_process.cnn.stacks[0].firstconv.layer)
        agent = MineRLConditionalAgent.__new__(MineRLConditionalAgent)
        agent.policy, agent.device = policy, torch.device('cuda')
        f = RLPolicy.__new__(RLPolicy)
        f.agent, f.torch, f.device, f.reward_mode = agent, torch, 'cuda', 'wood-potential-v4'
        metrics = OnsetMetrics(mapper.BUTTON_IDX_TO_FACTORED, mapper.BUTTON_IDX_TO_CAMERA_META_OFF,
                               mapper.CAMERA_IDX_TO_FACTORED, Buttons.ALL, mapper.camera_null_bin, 'cuda')
        baseline_error = 0.
        prefix_frames = branch_frames = 0
        for session in data['plan']['sessions']:
            check_stop()
            name = session['id']
            print('condition branches '+name, flush=True)
            rows = groups[name]
            first = next(i for i, row in enumerate(rows) if row['loss_eligible'])
            assert all(row['loss_eligible'] for row in rows[first:first+8]) and len(rows) >= first+8
            source = root/session['run']/'correction-rows.json'
            freeze(source, session['rows_sha256'])
            original = read(source)

            def screen_at(index):
                path = root/session['run']/original[index]['frame']
                assert digest(path) == session['frame_sha256'][index]
                with Image.open(path) as im:
                    return np.asarray(im.convert('RGB'))

            agent.hidden_state = device_tree(policy.initial_state(2), 'cuda')
            f.goal_embed = embeddings['original_wood']
            for i in range(first):
                check_stop()
                f.first = i == 0
                f.set_reward_context(original[i]['reward_context'])
                _, memory = f.features(screen_at(i))
                agent.hidden_state = memory
                prefix_frames += 1
            onset_memory = cpu_tree(agent.hidden_state)
            onset_hash = state_fingerprint(onset_memory)
            screens = [screen_at(i) for i in range(first, first+8)]
            for condition, embed in embeddings.items():
                agent.hidden_state = device_tree(cpu_tree(onset_memory), 'cuda')
                assert state_fingerprint(agent.hidden_state) == onset_hash
                f.goal_embed = embed
                measured = []
                for offset, screen in enumerate(screens):
                    check_stop()
                    i = first+offset
                    f.first = False
                    f.set_reward_context(original[i]['reward_context'])
                    features, memory = f.features(screen)
                    with torch.no_grad():
                        measured.append(metrics.measure(guided(actor, features), rows[i]['action']))
                    agent.hidden_state = memory
                    branch_frames += 1
                assert state_fingerprint(onset_memory) == onset_hash, 'branch mutated common memory'
                if condition == 'original_wood':
                    expected = training['after']['first_actions'][name]
                    error = max(abs(value-expected[key]) for key, value in measured[0].items())
                    baseline_error = max(baseline_error, error)
                    assert error < 1e-4, (name, error)
                report['rows'].append(dict(session=name, split=session['split'], condition=condition,
                                           onset_frame=first, onset_memory_sha256=onset_hash,
                                           first=measured[0], first8=aggregate(measured), per_frame=measured))
            write_json(out/'report.json', report)
        assert len(report['rows']) == 13*7 and prefix_frames == 7652 and branch_frames == 13*7*8
        for name, sha in report['frozen_sha256'].items():
            assert digest(root/name) == sha, name
        check_stop()
        report.update(completed=True, prefix_frames=prefix_frames, branch_frames=branch_frames,
                      original_baseline_max_metric_error=baseline_error, sources_unchanged=True,
                      checkpoint_sha256=digest(checkpoint_path),
                      limitation='Recorded teacher observations; no autonomous recovery or acquisition evidence.')
    except BaseException as exc:
        report['error'] = repr(exc)
        (out/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report.update(elapsed_s=time.monotonic()-started,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write_json(out/'report.json', report)
        publish_completion(stop, 'RL training ended; goal conditioning diagnosis finished; inspect report.')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows', 'frozen_sha256')}), flush=True)
    return 0 if report['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
