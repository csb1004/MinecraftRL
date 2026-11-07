"""Check that an untrained ReacquireHead leaves the official STEVE-1 action distribution unchanged (GPU, no game).

Runs the official policy over 64 recorded frames with the `mine iron ore` embedding and guidance 6, as the pipeline
does (CUDA autocast), once with the official head and once with a zero-initialized ReacquireHead whose stage input
changes every frame. Both runs start from the same initial state. Writes <out>/check.json; prints `identical: true`
only when every distribution key matches exactly.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

FRAMES = 'results/iron-demo-probe-20261104-arena-v2/iron_ore-mine_iron_text/trial-2/frames'
EMBED = 'results/iron-demo-prompts/mine_iron_ore.npy'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--frames', type=int, default=64)
    args = p.parse_args(argv)
    root = args.root.resolve(); out = (args.out or root / 'results/reacquire-head-check-20261107').resolve()
    import torch
    from PIL import Image
    sys.path.insert(0, str(root / 'external/steve1')); sys.path.insert(0, str(root / 'src'))
    from gym3.types import DictType
    from steve1.VPT.agent import AGENT_RESOLUTION, CameraHierarchicalMapping, resize_image
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    from minecraftrl.reacquire_head import ReacquireHead, STAGE_DIM

    with open(root / 'weights/steve1/2x.model', 'rb') as stream: parameters = pickle.load(stream)
    kwargs = parameters['model']['args']['net']['args']
    head_opts = dict(parameters['model']['args']['pi_head_opts']); head_opts['temperature'] = float(head_opts['temperature'])
    space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    policy = MinecraftAgentPolicy(policy_kwargs=kwargs, pi_head_kwargs=head_opts, action_space=space)
    policy.load_state_dict(torch.load(str(root / 'weights/steve1/steve1.weights'), map_location='cpu'), strict=True)
    policy = policy.cuda().eval()
    embed = np.load(root / EMBED)
    paths = sorted((root / FRAMES).glob('*.jpg'))[:args.frames]
    frames = [resize_image(np.asarray(Image.open(x).convert('RGB')), AGENT_RESOLUTION) for x in paths]

    def run():
        # Same shapes as MineRLConditionalAgent with guidance: two-row state, one-element first flag.
        state = policy.initial_state(2); first = torch.zeros((1,), dtype=torch.bool).cuda(); pds = []
        for i, frame in enumerate(frames):
            if isinstance(policy.pi_head, ReacquireHead): policy.pi_head.stage = i % STAGE_DIM
            obs = {'img': torch.from_numpy(frame[None]).cuda(), 'mineclip_embed': torch.from_numpy(embed).cuda()}
            action = {'buttons': torch.zeros((1, 1), dtype=torch.long).cuda(), 'camera': torch.zeros((1, 1), dtype=torch.long).cuda()}
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, state, result = policy.act(obs, first, state, stochastic=False, taken_action=action, return_pd=True,
                                              cond_scale=6.)
            pds.append({k: v.float().cpu() for k, v in result['pd'].items()})
        return pds

    official = run()
    policy.pi_head = ReacquireHead(policy.pi_head, policy.net.output_latent_size(), use_stage=True).cuda().eval()
    wrapped = run()
    diff = {k: max(float((a[k] - b[k]).abs().max()) for a, b in zip(official, wrapped)) for k in official[0]}
    report = dict(frames=len(frames), source=FRAMES, embedding=EMBED, cond_scale=6., autocast=True,
                  max_abs_diff=diff, identical=all(v == 0. for v in diff.values()))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report)); print('identical: %s' % str(report['identical']).lower())
    return 0 if report['identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
