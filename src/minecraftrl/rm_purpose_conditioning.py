"""P2 experiment admission for generic goal contracts; no new learned skill."""
import hashlib
import io
from pathlib import Path

from .goal_contract import compile_conditions, digest, read_json
from .rm_conditioning import ConditioningPolicy

MODES = ('official', 'purpose-fixed', 'purpose-switch')
OFFICIAL_PROMPT = 'chop down the tree, gather wood, pick up wood, chop it down, break tree'
CONTEXT = dict(schema=1, root_goal='collect at least one wood log',
    active_goal='collect at least one wood log', constraints=[],
    actions=dict(find='find', gather='collect'), targets=dict(source='a tree', resource='wood'),
    outcomes=dict(located=dict(description='a tree is located', predicate_ref='wood_v3_target_observed'),
                  acquired=dict(description='wood is added to the inventory', predicate_ref='server_inventory_wood_gain')),
    states=['seeking', 'gathering'], terminal_states=['complete'])


def candidate(mode):
    if mode not in MODES[1:]:
        raise ValueError('only contract modes have compiled candidates')
    conditions = [dict(id='gather', action_ref='gather', target_ref='resource', outcome_ref='acquired')]
    if mode == 'purpose-switch':
        conditions.insert(0, dict(id='seek', action_ref='find', target_ref='source', outcome_ref='located'))
    proposal = dict(schema=1, context_sha256=digest(CONTEXT), status='candidate', reason='',
        conditions=conditions, state_bindings=[
            dict(state='seeking', condition_id='seek' if mode == 'purpose-switch' else 'gather'),
            dict(state='gathering', condition_id='gather')])
    return compile_conditions(CONTEXT, proposal)


def prompts():
    return dict(root=OFFICIAL_PROMPT, **{
        key: value['prompt'] for key, value in candidate('purpose-switch')['conditions'].items()})


def checked_tokens(prompt, tokenizer, tokenize_batch):
    """Reject truncation before official MineCLIP encoding, including BOS/EOS."""
    ids = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    complete = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    if len(complete) > 77:
        raise ValueError('prompt exceeds MineCLIP context; explicit revision required')
    padded = complete + [0] * (77 - len(complete))
    actual = tokenize_batch(prompt, max_length=77, language_model='clip')[0].tolist()
    if actual != padded:
        raise ValueError('tokenizer path differs from checked full prompt')
    return dict(count_including_special=len(complete), ids=actual, sha256=digest(actual), truncated=False)


def load_bank(path):
    import numpy as np
    path = Path(path).resolve()
    meta = read_json(path/'manifest.json')
    expected_candidates = {mode:candidate(mode) for mode in MODES[1:]}
    if (meta.get('schema') != 'purpose_conditioning_p2_v1' or meta.get('seed') != 31031
            or meta.get('prompts') != prompts() or meta.get('candidates') != expected_candidates
            or meta.get('capability_verified') is not False):
        raise ValueError('goal bank does not match locked P2 contracts')
    if set(meta.get('embeddings', {})) != set(prompts()) or set(meta.get('tokens', {})) != set(prompts()):
        raise ValueError('missing P2 embeddings/token receipts')
    embeddings = {}
    for key in prompts():
        token = meta['tokens'][key]
        ids = token.get('ids', [])
        count = token.get('count_including_special')
        if (type(count) is not int or not 2 <= count <= 77 or len(ids) != 77
                or token.get('truncated') is not False or token.get('sha256') != digest(ids)):
            raise ValueError('invalid token receipt')
        raw = (path/(key+'.npy')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['embeddings'][key]['file_sha256']:
            raise ValueError('goal file hash mismatch')
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if value.shape != (1,512) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid goal vector')
        if hashlib.sha256(value.tobytes()).hexdigest() != meta['embeddings'][key]['vector_sha256']:
            raise ValueError('goal vector hash mismatch')
        embeddings[key] = value
    return embeddings, meta


def next_phase(phase, confirmed_stage):
    if confirmed_stage == 'complete':
        return 'complete'
    if phase == 'complete':
        raise RuntimeError('cannot resume completed condition')
    return 'gathering' if phase == 'gathering' or confirmed_stage != 'search' else 'seeking'


def condition_key(mode, phase):
    if phase == 'complete':
        raise InterruptedError('no condition after completion')
    if phase not in CONTEXT['states'] or mode not in MODES:
        raise ValueError('unknown P2 mode/phase')
    return 'root' if mode == 'official' else candidate(mode)['state_bindings'][phase]


class PurposeConditioningPolicy(ConditioningPolicy):
    bank_loader = staticmethod(load_bank)

    def __init__(self, *args, **kwargs):
        self.phase = 'seeking'
        super().__init__(*args, **kwargs)

    def observe_stage(self, stage):
        super().observe_stage(stage)
        self.phase = next_phase(self.phase, self.selector.current)

    def goal_key(self):
        return condition_key(self.mode, self.phase)

    def row_context(self):
        return dict(execution_phase=self.phase, context_sha256=digest(CONTEXT))
