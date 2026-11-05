"""Freeze demo text-prompt embeddings (official STEVE-1 prior); no gameplay."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

from minecraftrl.runtime import write_json

PROMPTS = {'mine_iron_ore': 'mine iron ore', 'break_iron_ore_block': 'break the iron ore block', 'mine_stone': 'mine stone'}
# Pit-escape candidates: 'build a tower' follows the STEVE-1 prompt-chaining example; the others name pillar and climb strategies.
ESCAPE_PROMPTS = {'build_a_tower': 'build a tower', 'pillar_jump': 'pillar jump up', 'climb_out': 'climb out of the hole'}
# Ceiling candidates: clear blocks over the head (the failed pillar trial sealed its own shaft with dirt).
CEILING_PROMPTS = {'dig_up': 'dig up', 'break_block_above': 'break the block above you', 'mine_ceiling': 'mine the ceiling'}
SETS = {'iron': PROMPTS, 'escape': ESCAPE_PROMPTS, 'ceiling': CEILING_PROMPTS}
SEED = 31031


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--set', choices=sorted(SETS), default='iron')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not args.output.resolve().is_relative_to(args.root / 'results'):
        raise ValueError('prompt embeddings must be under project results')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / 'external/steve1'))
    import numpy as np
    import torch
    from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
    from steve1.mineclip_code.load_mineclip import load
    from steve1.data.text_alignment.vae import load_vae_model
    from steve1.utils.embed_utils import get_prior_embed
    config = copy.deepcopy(MINECLIP_CONFIG)
    config['ckpt']['path'] = str(args.root / 'weights/steve1/attn.pth')
    model = load(config, device='cuda').eval()
    prior = load_vae_model(dict(PRIOR_INFO, model_path=str(args.root / 'weights/steve1/steve1_prior.pt')))
    embeddings = {}
    for key, prompt in SETS[args.set].items():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            value = get_prior_embed(prompt, model, prior, 'cuda').astype(np.float32)
        if value.shape != (1, 512) or not np.isfinite(value).all():
            raise ValueError('invalid generated prompt embedding')
        path = args.output / (key + '.npy')
        np.save(path, value, allow_pickle=False)
        embeddings[key] = dict(prompt=prompt, file=path.relative_to(args.root).as_posix(),
                               file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               vector_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    from minecraftrl.artifact_hash import sha256
    weights = {name: sha256(args.root / 'weights/steve1' / name)
               for name in ('attn.pth', 'steve1_prior.pt', 'steve1.weights', '2x.model')}
    visual = {name: dict(file='weights/steve1/visual-prompts/' + name,
                         file_sha256=hashlib.sha256((args.root / 'weights/steve1/visual-prompts' / name).read_bytes()).hexdigest())
              for name in ('dig.npy', 'wood.npy')}
    write_json(args.output / 'manifest.json', dict(schema='iron_demo_prompts_v1', prompt_set=args.set, seed=SEED, embeddings=embeddings,
               official_visual_candidates=visual, official_weights=weights,
               peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
               scope='condition preparation only; gameplay and optimizer updates zero'))
    print('Prepared %d text-prompt embeddings; no gameplay.' % len(embeddings))


if __name__ == '__main__':
    main()
