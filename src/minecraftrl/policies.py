"""Pixel-input adapters around pinned official code. Recurrent state stays here."""
import gc
import hashlib
import math
import pickle
import sys
from pathlib import Path
from typing import Protocol, Optional, Any


class Policy(Protocol):
    def reset(self, goal: Optional[str] = None) -> None: ...
    def act(self, screen: Any) -> dict: ...
    def close(self) -> None: ...


def steve_guidance(value):
    if isinstance(value,bool):raise ValueError('STEVE-1 guidance must be a finite number in [0, 7]')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=7:
        raise ValueError('STEVE-1 guidance must be a finite number in [0, 7]')
    return value


def model_kwargs(path):
    # Only artifacts whose download receipt was checked by the launcher reach here.
    with open(path, 'rb') as stream: parameters = pickle.load(stream)
    policy = parameters['model']['args']['net']['args']
    head = dict(parameters['model']['args']['pi_head_opts'])
    head['temperature'] = float(head['temperature'])
    return policy, head


class VPTPolicy:
    def __init__(self, env, root, device='cuda'):
        import torch
        self.torch = torch; self.device = device
        sys.path.insert(0, str(Path(root) / 'external' / 'vpt'))
        from agent import MineRLAgent
        kwargs, head = model_kwargs(Path(root) / 'weights/vpt/foundation-model-1x.model')
        self.agent = MineRLAgent(env, device=device, policy_kwargs=kwargs, pi_head_kwargs=head)
        state = torch.load(str(Path(root) / 'weights/vpt/foundation-model-1x.weights'), map_location=device)
        self.agent.policy.load_state_dict(state, strict=True)
        self.agent.policy.eval()
        del state

    def reset(self, goal=None):
        if goal is not None: raise ValueError('VPT foundation is not a natural-language conditioned policy')
        self.agent.reset()

    def act(self, screen):
        with self.torch.no_grad(): return self.agent.get_action({'pov': screen})

    def close(self):
        self.agent = None; gc.collect()
        if self.torch.cuda.is_available(): self.torch.cuda.empty_cache()


class Steve1Policy:
    # Pinned upstream steve1/run_agent/paper_prompts.py; opt-in comparison profile.
    OFFICIAL_PROMPTS={
        'collect dirt':'get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt',
        'collect wood':'chop down the tree, gather wood, pick up wood, chop it down, break tree',
    }
    # Official download_weights.sh at 903b244796322f4d0073a8f62c05f51eac3aed52.
    # Converted without numeric changes from the published visual prompt arrays.
    OFFICIAL_VISUAL={
        'dig':('dig.npy','ba1fbe4516e3804ca270d3327c1dbfaed9589b5027bb8c554b76e73a22ffaa47'),
        'collect wood':('wood.npy','768bb35ed2914b3e89769acc4bfee82889e2db2f9909d395a7ef36899366827c'),
        'collect dirt':('dirt.npy','cd15f2c913b3ef2014c6112ff0d0c944472c3addc9cf7988bba539812bceab25'),
    }

    def __init__(self, env, root, device='cuda',prompt_profile='short',guidance=None):
        if prompt_profile not in ('short','official','official-visual'): raise ValueError('unknown STEVE-1 prompt profile')
        self.guidance=None if guidance is None else steve_guidance(guidance)
        self.prompt_profile=prompt_profile; self.condition_metadata=None
        import torch
        self.torch = torch; self.device = device; self.root = Path(root); self.goal_embed = None
        sys.path.insert(0, str(self.root / 'external/steve1'))
        from steve1.MineRLConditionalAgent import MineRLConditionalAgent
        from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
        from steve1.mineclip_code.load_mineclip import load
        from steve1.data.text_alignment.vae import load_vae_model
        from steve1.utils.embed_utils import get_prior_embed
        kwargs, head = model_kwargs(self.root / 'weights/steve1/2x.model')
        self.agent = MineRLConditionalAgent(env, device=device, policy_kwargs=kwargs, pi_head_kwargs=head)
        state = torch.load(str(self.root / 'weights/steve1/steve1.weights'), map_location=device)
        self.agent.policy.load_state_dict(state, strict=True); self.agent.policy.eval(); del state
        if prompt_profile=='official-visual':
            self.mineclip=self.prior=self.embed=None
            return
        import copy
        clip_config = copy.deepcopy(MINECLIP_CONFIG)
        clip_config['ckpt']['path'] = str(self.root / 'weights/steve1/attn.pth')
        self.mineclip = load(clip_config, device=device); self.mineclip.eval()
        prior_config = dict(PRIOR_INFO, model_path=str(self.root / 'weights/steve1/steve1_prior.pt'))
        self.prior = load_vae_model(prior_config); self.embed = get_prior_embed

    def reset(self, goal=None):
        if not goal or not goal.strip(): raise ValueError('STEVE-1 requires a text goal')
        scale=getattr(self,'guidance',None)
        if scale is None:scale=7. if self.prompt_profile=='official-visual' else 6.
        if self.prompt_profile=='official-visual':
            import io
            import numpy as np
            if goal not in self.OFFICIAL_VISUAL:raise ValueError('unsupported official visual collection goal')
            name,expected=self.OFFICIAL_VISUAL[goal]
            relative=Path('weights/steve1/visual-prompts')/name
            path=self.root/relative
            if path.stat().st_size>16384:raise ValueError('oversized official visual embedding')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('official visual embedding hash mismatch')
            value=np.load(io.BytesIO(raw),allow_pickle=False)
            if value.shape!=(1,512) or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid official visual embedding')
            self.agent.reset(cond_scale=scale);self.goal_embed=value
            self.condition_metadata=dict(profile=self.prompt_profile,requested_goal=goal,effective_prompt=None,
                cond_scale=scale,embedding_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                source='official_video_embedding',source_file=relative.as_posix(),source_file_sha256=expected)
            return
        if self.prompt_profile=='official':
            if goal not in self.OFFICIAL_PROMPTS: raise ValueError('unsupported official collection goal')
            prompt=self.OFFICIAL_PROMPTS[goal]
        else: prompt=goal
        self.agent.reset(cond_scale=scale)
        self.goal_embed = self.embed(prompt, self.mineclip, self.prior, self.device)
        self.condition_metadata={'profile':self.prompt_profile,'requested_goal':goal,'effective_prompt':prompt,
                                 'cond_scale':scale,'embedding_sha256':hashlib.sha256(self.goal_embed.tobytes()).hexdigest()}

    def act(self, screen):
        if self.goal_embed is None: raise RuntimeError('reset(goal) must precede act')
        with self.torch.no_grad(), self.torch.cuda.amp.autocast():
            return self.agent.get_action({'pov': screen}, self.goal_embed)

    def close(self):
        self.agent = self.mineclip = self.prior = self.goal_embed = None; gc.collect()
        if self.torch.cuda.is_available(): self.torch.cuda.empty_cache()


class MineRLEnvironment:
    def __init__(self, raw):
        self.raw = raw; self.terminal = False; self.initialized = False; self.last_observation = None
        self.last_info = {}
    def seed(self, seed): return self.raw.seed(seed)
    def reset(self):
        observation = self.raw.reset(); self.terminal = False; self.initialized = True
        self.last_info = {}
        self.last_observation = observation
        return observation
    def step(self, action):
        transition = self.raw.step(action); self.terminal = bool(transition[2])
        self.last_info = transition[3]
        self.last_observation = transition[0]
        return transition
    def release(self):
        if not self.initialized: return 'environment_not_initialized_no_policy_input'
        if self.terminal: return 'environment_done_no_neutral_step'
        # Neutral step is cleanup, not a policy action; terminal MineRL rejects it.
        transition=self.step(self.raw.action_space.noop())
        server=transition[3].get('workshop',{})
        self.release_evidence={k:server.get(k) for k in ('server_available','server_tick','held_key_count','server_alive','error')}
        self.release_evidence['verified']=(server.get('server_available')==1 and not server.get('error') and server.get('held_key_count')==0)
        return 'neutral_command_sent'
    def close(self): return self.raw.close()


class PersistentMineRLEnvironment(MineRLEnvironment):
    """Use upstream save-and-quit before taking a snapshot; never copy a live world."""
    def __init__(self,raw,minecraft_dir,temporary_dir):
        super().__init__(raw)
        self.minecraft_dir=Path(minecraft_dir); self.temporary_dir=Path(temporary_dir)
        self.saved_world=None

    def world_folders(self):
        return {p.parent.resolve() for p in (self.minecraft_dir/'saves').glob('*/level.dat')} | {
            p.parent.resolve() for p in self.temporary_dir.glob('*/saves/*/level.dat')}

    def reset(self):
        before=self.world_folders()
        observation=super().reset()
        created=self.world_folders()-before
        if len(created)!=1: raise RuntimeError('Cannot identify one owned Minecraft world')
        self.saved_world=created.pop()
        return observation

    def save_and_quit(self):
        if not self.initialized or self.saved_world is None: raise RuntimeError('World not initialized')
        if len(self.raw.instances)!=1: raise RuntimeError('Persistence requires one local agent')
        instance=self.raw.instances[0]; socket=instance.client_socket
        timeout=socket.gettimeout()
        try:
            socket.settimeout(8.)
            # The pinned upstream protocol returns only after save/unload reaches
            # the main menu. raw.close alone kills Java and is not a save barrier.
            self.raw._TO_MOVE_quit_current_episode(instance)
        finally: socket.settimeout(timeout)
        self.initialized=False
        return self.saved_world
