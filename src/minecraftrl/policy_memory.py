"""Bounded, non-pickle recurrent/RNG snapshots for the same saved-world task."""
from copy import deepcopy
import hashlib
from pathlib import Path
import zipfile
import numpy as np

MAX_MEMORY_BYTES=128*1024**2


def encode_state(state):
    arrays={}
    def visit(value,depth=0):
        if depth>12:raise ValueError('policy memory nesting limit')
        if value is None:return {'kind':'none'}
        if isinstance(value,(tuple,list)):
            return dict(kind='tuple' if isinstance(value,tuple) else 'list',items=[visit(v,depth+1) for v in value])
        if not hasattr(value,'detach'):raise ValueError('unexpected policy memory node')
        array=value.detach().cpu().contiguous().numpy().copy()
        if array.dtype.kind not in 'bifu' or not np.isfinite(array).all():raise ValueError('invalid policy memory tensor')
        if len(arrays)>=64:raise ValueError('policy memory tensor limit')
        key='state_%02d'%len(arrays);arrays[key]=array
        return dict(kind='tensor',key=key,shape=list(array.shape),dtype=str(array.dtype))
    tree=visit(state)
    if sum(v.nbytes for v in arrays.values())>MAX_MEMORY_BYTES:raise ValueError('policy memory size limit')
    return tree,arrays


def decode_state(tree,arrays,torch,device):
    kind=tree['kind']
    if kind=='none':return None
    if kind in ('list','tuple'):
        values=[decode_state(v,arrays,torch,device) for v in tree['items']]
        return tuple(values) if kind=='tuple' else values
    value=arrays[tree['key']]
    if list(value.shape)!=tree['shape'] or str(value.dtype)!=tree['dtype'] or not np.isfinite(value).all():
        raise ValueError('policy memory tensor mismatch')
    return torch.from_numpy(value.copy()).to(device)


def memory_identity(policy,root):
    from .world_store import digest
    import subprocess
    root=Path(root)
    paths=['weights/steve1/steve1.weights','weights/steve1/2x.model',
           'external/steve1/steve1/MineRLConditionalAgent.py',
           'external/steve1/steve1/embed_conditioned_policy.py','src/minecraftrl/policies.py']
    return dict(torch=str(policy.torch.__version__),files={p:digest(root/p) for p in paths},
        upstream_commit=subprocess.check_output(['git','-C',str(root/'external/steve1'),'rev-parse','HEAD'],text=True).strip())


def initialized_template(policy):
    """One synthetic inference before control, restoring all original state and RNG."""
    torch=policy.torch;agent=policy.agent;device=policy.device
    hidden=agent.hidden_state;scale=agent.cond_scale
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(device)
    try:
        agent.reset(cond_scale=7.)
        with torch.no_grad(),torch.cuda.amp.autocast():
            agent.get_action({'pov':np.zeros((360,640,3),np.uint8)},np.zeros((1,512),np.float32))
        return encode_state(agent.hidden_state)[0]
    finally:
        agent.hidden_state=hidden;agent.cond_scale=scale
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,device)


class PolicyMemory:
    def __init__(self,policy,identity,template,bundle=None):
        self.policy=policy;self.identity=identity;self.template=template
        self.source=deepcopy((bundle or {}).get('policy_memory'))
        self.path=(bundle or {}).get('policy_memory_path');self.arrays=None
        self.context=None;self.consumed=False;self.restored=False;self.inferences=0
        self.receipt=dict(enabled=True,restored=False,reason='no_saved_memory')
        if self.source:
            from .world_store import digest
            if (self.source['identity']!=identity or self.source['tree']!=template
                    or self.source['world_archive_sha256']!=bundle['archive_sha256']):
                raise ValueError('incompatible saved policy memory')
            path=Path(self.path)
            if path.is_symlink() or path.stat().st_size>MAX_MEMORY_BYTES or digest(path)!=self.source['file_sha256']:
                raise ValueError('policy memory file mismatch')
            with zipfile.ZipFile(path) as archive:
                entries=archive.infolist()
                if (len(entries)>66 or len({e.filename for e in entries})!=len(entries)
                        or sum(e.file_size for e in entries)>MAX_MEMORY_BYTES):
                    raise ValueError('policy memory expanded size limit')
            with np.load(path,allow_pickle=False) as stored:
                self.arrays={k:stored[k].copy() for k in stored.files}
            expected=set()
            def keys(node):
                if node['kind']=='tensor':expected.add(node['key'])
                for child in node.get('items',[]):keys(child)
            keys(template)
            if set(self.arrays)!=expected|{'rng_cpu','rng_device'}:raise ValueError('policy memory array keys mismatch')
            for key in ('rng_cpu','rng_device'):
                if self.arrays[key].dtype!=np.uint8 or self.arrays[key].ndim!=1:
                    raise ValueError('invalid policy RNG state')
            # Shape/finite validation is before any tensor reaches the GPU.
            from .rl_policy import state_fingerprint
            state=decode_state(template,self.arrays,policy.torch,'cpu')
            if state_fingerprint(state)!=self.source['memory_sha256']:
                raise ValueError('policy memory fingerprint mismatch')
            self.receipt.update(reason='awaiting_task_match',source_file_sha256=self.source['file_sha256'])

    def on_reset(self,task,world_id):
        condition=self.policy.condition_metadata
        self.context=dict(world_id=world_id,task_id=task['task_id'],revision=task['revision'],goal=task['goal'],
            root_embedding_sha256=condition['embedding_sha256'],root_cond_scale=condition['cond_scale'])
        self.inferences=0;self.restored=False
        if self.consumed:return dict(self.receipt,restored=False,reason='later_task_reset')
        self.consumed=True
        if self.source and self.source['context']==self.context:
            torch=self.policy.torch;device=self.policy.device
            state=decode_state(self.template,self.arrays,torch,device)
            cpu=torch.from_numpy(self.arrays['rng_cpu'].copy())
            rng=torch.from_numpy(self.arrays['rng_device'].copy())
            if cpu.shape!=torch.get_rng_state().shape:raise ValueError('policy CPU RNG shape mismatch')
            current=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else torch.get_rng_state()
            if rng.shape!=current.shape:raise ValueError('policy device RNG shape mismatch')
            self.policy.agent.hidden_state=state;torch.set_rng_state(cpu)
            if str(device).startswith('cuda'):torch.cuda.set_rng_state(rng,device)
            from .rl_policy import state_fingerprint
            fingerprint=state_fingerprint(state)
            if fingerprint!=self.source['memory_sha256']:raise ValueError('policy memory fingerprint mismatch')
            self.restored=True
            self.receipt.update(restored=True,reason='same_world_task',memory_sha256=fingerprint,
                game_physics_bit_exact=False,policy_actions_during_restore=0)
        elif self.source:self.receipt.update(restored=False,reason='task_or_condition_changed')
        return deepcopy(self.receipt)

    def inferred(self):self.inferences+=1

    def snapshot(self,output):
        if self.context is None:
            return dict(path=self.path,metadata=self.source) if self.source else None
        if not self.inferences and not self.restored:return None
        from .rl_policy import state_fingerprint
        torch=self.policy.torch;device=self.policy.device;agent=self.policy.agent
        tree,arrays=encode_state(agent.hidden_state)
        if tree!=self.template:raise ValueError('live policy memory structure changed')
        arrays['rng_cpu']=torch.get_rng_state().cpu().numpy().copy()
        rng=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else torch.get_rng_state()
        arrays['rng_device']=rng.cpu().numpy().copy()
        path=Path(output)/'policy-memory.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        metadata=dict(schema=1,identity=self.identity,tree=tree,context=self.context,
            memory_sha256=state_fingerprint(agent.hidden_state),source_run=str(output),
            inferred_frames=self.inferences,includes_undispatched_inference=True,
            active_embedding_sha256=hashlib.sha256(self.policy.goal_embed.tobytes()).hexdigest(),
            active_cond_scale=agent.cond_scale)
        return dict(path=str(path),metadata=metadata)
