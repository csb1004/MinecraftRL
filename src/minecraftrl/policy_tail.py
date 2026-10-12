"""Explicit pretrained tail checkpoint contract; legacy heads remain unchanged."""
NAMES=('lastlayer','final_ln')
TEMPORAL_SCOPE='head_tail_last_recurrent'


def tail_state(net):
    return {name:{k:v.detach().cpu().clone() for k,v in getattr(net,name).state_dict().items()} for name in NAMES}


def temporal_state(net):
    blocks=net.recurrent_layer.blocks
    return dict(index=len(blocks)-1,count=len(blocks),
                state={k:v.detach().cpu().clone() for k,v in blocks[-1].state_dict().items()})


def validate_tensors(saved,module):
    import torch
    expected=module.state_dict()
    if not isinstance(saved,dict) or set(saved)!=set(expected):raise ValueError('backbone keys mismatch')
    for k,v in saved.items():
        if (not isinstance(v,torch.Tensor) or v.shape!=expected[k].shape or
                v.dtype!=expected[k].dtype or not torch.isfinite(v).all()):
            raise ValueError('invalid backbone tensor')


def load_tail(net,checkpoint):
    saved=checkpoint.get('backbone_tail')
    temporal=checkpoint.get('backbone_temporal')
    scope=checkpoint.get('scope')
    if scope!=TEMPORAL_SCOPE and temporal is not None:raise ValueError('temporal weights in incompatible scope')
    if scope not in ('head_and_tail',TEMPORAL_SCOPE):
        if saved is not None:raise ValueError('tail weights in incompatible checkpoint scope')
        return False
    if not isinstance(saved,dict) or set(saved)!=set(NAMES):raise ValueError('missing backbone tail')
    # Validate every tensor before mutating any module.
    for name in NAMES:
        validate_tensors(saved[name],getattr(net,name))
    if scope==TEMPORAL_SCOPE:
        blocks=net.recurrent_layer.blocks
        if (not isinstance(temporal,dict) or set(temporal)!={'index','count','state'} or
                type(temporal['count']) is not int or type(temporal['index']) is not int or
                temporal['count']!=len(blocks) or temporal['index']!=len(blocks)-1):
            raise ValueError('invalid temporal block identity')
        validate_tensors(temporal['state'],blocks[-1])
    for name in NAMES:getattr(net,name).load_state_dict(saved[name],strict=True)
    if scope==TEMPORAL_SCOPE:blocks[-1].load_state_dict(temporal['state'],strict=True)
    return True
