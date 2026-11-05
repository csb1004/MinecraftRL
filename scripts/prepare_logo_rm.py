"""Review a generated validator proposal; optionally encode new policy conditions.

No game, model training, API call, STOP change, or mutation of previous results.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from minecraftrl.goal_contract import read_json
from minecraftrl.logo_validators import request, ValidatorSet
from minecraftrl.logo_wood import context, acceptance_cases, action_candidate, prompts


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not output.is_relative_to(root/'results'):
        raise ValueError('output must be under project results')
    output.mkdir(parents=True, exist_ok=False)
    proposal = read_json(args.proposal)
    save(output/'request.json', request(context()))
    save(output/'proposal.json', proposal)
    save(output/'acceptance-cases.json', acceptance_cases())
    try:
        validators = ValidatorSet(context(), proposal)
        review = validators.verify(acceptance_cases())
    except ValueError as exc:
        save(output/'review.json', dict(passed=False, compile_error=str(exc), context=request(context())))
        raise
    save(output/'review.json', review)
    if not review['passed']:
        raise ValueError('rejected proposal; review.json is revision feedback for its author')
    save(output/'policy-candidate.json', action_candidate())
    files = [*sorted((root/'src/minecraftrl').glob('logo_*.py')), root/'src/minecraftrl/goal_actions.py',
             root/'src/minecraftrl/rm_logo_conditioning.py', root/'src/minecraftrl/rm_conditioning.py',
             root/'src/minecraftrl/cli.py', Path(__file__).resolve()]
    save(output/'provenance.json', dict(author='Codex in this conversation',
        source='LOGO publisher abstract/intro; Minecraft adaptation, not official LOGO code',
        source_url='https://www.sciencedirect.com/science/article/pii/S0925231226016899',
        API_calls=0, independent_LLM_review=False, observer_changed=False,
        files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print('Caller-owned validator cases passed; no policy capability claim.', flush=True)
    if not args.encode:
        return
    sys.path.insert(0, str(root/'external/steve1'))
    import numpy as np
    import torch
    from mineclip.mineclip.tokenization import get_tokenizer, tokenize_batch
    from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
    from steve1.mineclip_code.load_mineclip import load
    from steve1.data.text_alignment.vae import load_vae_model
    from steve1.utils.embed_utils import get_prior_embed
    from minecraftrl.rm_purpose_conditioning import checked_tokens
    from minecraftrl.rm_logo_conditioning import load_bank
    from minecraftrl.artifact_hash import sha256
    tokens = {key:checked_tokens(prompt, get_tokenizer('clip'), tokenize_batch) for key,prompt in prompts().items()}
    save(output/'checked-prompts.json', dict(prompts=prompts(), tokens=tokens))
    config = copy.deepcopy(MINECLIP_CONFIG)
    config['ckpt']['path'] = str(root/'weights/steve1/attn.pth')
    model = load(config, device='cuda').eval()
    prior = load_vae_model(dict(PRIOR_INFO, model_path=str(root/'weights/steve1/steve1_prior.pt')))
    embeddings = {}
    for key,prompt in prompts().items():
        torch.manual_seed(31031)
        torch.cuda.manual_seed_all(31031)
        with torch.no_grad():
            value = get_prior_embed(prompt, model, prior, 'cuda').astype(np.float32)
        if value.shape != (1,512) or not np.isfinite(value).all():
            raise ValueError('invalid generated goal')
        path = output/(key+'.npy')
        with path.open('xb') as handle:
            np.save(handle, value, allow_pickle=False)
        embeddings[key] = dict(file_sha256=sha256(path), vector_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    weights = {name:sha256(root/'weights/steve1'/name) for name in ('attn.pth','steve1_prior.pt','steve1.weights','2x.model')}
    save(output/'manifest.json', dict(schema='logo_conditioning_v1', context=context(), proposal=proposal, review=review,
        validator_sha256=validators.sha256, candidate=action_candidate(), prompts=prompts(), tokens=tokens,
        seed=31031, embeddings=embeddings, official_weights=weights, capability_verified=False,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20, scope='preparation only; gameplay and RL updates zero'))
    load_bank(output)
    print('Prepared and reloaded two new condition embeddings; no gameplay or RL updates.', flush=True)


if __name__ == '__main__':
    main()
