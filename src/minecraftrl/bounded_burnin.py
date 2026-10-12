"""Exact last-attention-block burn-in over its finite K/V cache only."""
from .temporal_bc import device_tree, latent_features, recurrent_step


def cache_length(block):
    from steve1.VPT.lib.masked_attention import MaskedAttention
    from steve1.VPT.lib.util import ResidualRecurrentBlock
    from steve1.VPT.lib.xf import All2All, SelfAttentionLayer
    if type(block) is not ResidualRecurrentBlock or block.recurrence_type != 'transformer':
        raise ValueError('finite-cache optimization requires the official residual transformer block')
    attention = block.r
    if type(attention) is not MaskedAttention or attention.mask != 'clipped_causal':
        raise ValueError('unsupported attention/reset semantics')
    inner = attention.orc_block
    if type(inner) is not SelfAttentionLayer or type(inner.attn) is not All2All:
        raise ValueError('unsupported attention state construction')
    length = attention.maxlen
    if type(length) is not int or length <= 0 or inner.cache_keep_len != length:
        raise ValueError('invalid finite key/value cache')
    return length


def forward_window_bounded(block, tail, rows, initial, start, end, *, check_stop=lambda: None):
    """Rows must be fixed inputs from the frozen earlier blocks, in one episode.

    Official cached keys/values depend on the current row and current weights,
    not on this block's earlier attention outputs. Replaying a full cache
    therefore replaces every stale entry/mask without replaying earlier rows.
    This does not shorten the history already encoded by the frozen prefix.
    """
    import torch
    if not 0 <= start < end <= len(rows):
        raise ValueError('invalid episode window')
    begin = max(0, start-cache_length(block))
    memory = device_tree(initial, next(block.parameters()).device)
    with torch.no_grad():
        for i in range(begin, start):
            check_stop()
            _, memory = recurrent_step(block, rows[i], memory, i)
    outputs = []
    for i in range(start, end):
        check_stop()
        latent, memory = recurrent_step(block, rows[i], memory, i)
        outputs.append(latent_features(tail, latent, rows[i]))
    return torch.cat(outputs, 1)
