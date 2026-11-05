"""Load a local experimental STEVE-1 condition without advertising a validated skill."""
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from .world_store import digest


def load_condition(path,root,*,kind='experimental_hazard_condition'):
    if kind not in ('experimental_hazard_condition','experimental_tool_condition'):
        raise ValueError('unsupported experimental condition scope')
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root/'results') or path.stat().st_size>16384:
        raise ValueError('bounded project-local hazard condition required')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    fields={'schema','kind','prompt','seed','path','file_sha256','embedding_sha256','base_model_sha256',
            'prior_sha256','mineclip_sha256','source_commit','validated_skill'}
    scale=manifest.get('cond_scale',6.)
    if (set(manifest)-{'cond_scale'}!=fields or manifest['schema']!=1 or manifest['kind']!=kind
            or type(scale) not in (int,float) or not math.isfinite(scale) or not 0<=scale<=7
            or manifest['validated_skill'] is not False or not isinstance(manifest['prompt'],str)
            or not 1<=len(manifest['prompt'])<=256 or type(manifest['seed']) is not int or not 0<=manifest['seed']<2**32
            or manifest['source_commit']!='903b244796322f4d0073a8f62c05f51eac3aed52'):
        raise ValueError('invalid experimental hazard condition')
    source=(root/manifest['path']).resolve()
    if not source.is_relative_to(root/'results') or source.stat().st_size>16384:
        raise ValueError('invalid hazard embedding path')
    blob=source.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=manifest['file_sha256']:
        raise ValueError('hazard embedding hash mismatch')
    value=np.load(io.BytesIO(blob),allow_pickle=False)
    if value.shape!=(1,512) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('invalid hazard embedding')
    if hashlib.sha256(value.tobytes()).hexdigest()!=manifest['embedding_sha256']:
        raise ValueError('hazard tensor hash mismatch')
    for name,key in [('steve1.weights','base_model_sha256'),('steve1_prior.pt','prior_sha256'),('attn.pth','mineclip_sha256')]:
        if digest(root/'weights/steve1'/name)!=manifest[key]:raise ValueError('hazard condition model mismatch')
    return dict(embedding=value,sha256=manifest['embedding_sha256'],cond_scale=float(scale),provenance=manifest)
