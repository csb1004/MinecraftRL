"""Freeze P2 contract prompts and check exact MineCLIP tokenization before encoding."""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

from minecraftrl.rm_purpose_conditioning import MODES, candidate, prompts, checked_tokens, load_bank
from minecraftrl.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.output.resolve().is_relative_to(root/'results'):
        raise ValueError('goal bank must be under project results')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'external/steve1'))
    import numpy as np
    import torch
    from mineclip.mineclip.tokenization import get_tokenizer, tokenize_batch
    from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
    from steve1.mineclip_code.load_mineclip import load
    from steve1.data.text_alignment.vae import load_vae_model
    from steve1.utils.embed_utils import get_prior_embed
    tokens = {key:checked_tokens(prompt, get_tokenizer('clip'), tokenize_batch)
              for key, prompt in prompts().items()}
    write_json(args.output/'checked-prompts.json', dict(prompts=prompts(), tokens=tokens))
    config = copy.deepcopy(MINECLIP_CONFIG)
    config['ckpt']['path'] = str(root/'weights/steve1/attn.pth')
    model = load(config, device='cuda').eval()
    prior = load_vae_model(dict(PRIOR_INFO, model_path=str(root/'weights/steve1/steve1_prior.pt')))
    embeddings = {}
    for key, prompt in prompts().items():
        torch.manual_seed(31031)
        torch.cuda.manual_seed_all(31031)
        with torch.no_grad():
            value = get_prior_embed(prompt, model, prior, 'cuda').astype(np.float32)
        if value.shape != (1,512) or not np.isfinite(value).all():
            raise ValueError('invalid generated goal')
        path = args.output/(key+'.npy')
        np.save(path, value, allow_pickle=False)
        embeddings[key] = dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               vector_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    from minecraftrl.artifact_hash import sha256
    weights = {name:sha256(root/'weights/steve1'/name)
               for name in ('attn.pth','steve1_prior.pt','steve1.weights','2x.model')}
    write_json(args.output/'manifest.json', dict(schema='purpose_conditioning_p2_v1',
        prompts=prompts(), tokens=tokens, seed=31031, embeddings=embeddings,
        candidates={mode:candidate(mode) for mode in MODES[1:]}, official_weights=weights,
        capability_verified=False, admission='P2 frozen local diagnostic only',
        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        scope='goal preparation only; gameplay and optimizer updates zero'))
    load_bank(args.output)
    print('Prepared three P2 embeddings with untruncated tokens; no gameplay or RL updates.', flush=True)


if __name__ == '__main__':
    main()
