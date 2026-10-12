"""Pixel-input adapters around pinned official code. Recurrent state stays here."""
import gc
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Protocol, Optional, Any


class Policy(Protocol):
    def reset(self, goal: Optional[str] = None) -> None: ...
    def act(self, screen: Any) -> dict: ...
    def close(self) -> None: ...


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

    def __init__(self, env, root, device='cuda',prompt_profile='short'):
        if prompt_profile not in ('short','official'): raise ValueError('unknown STEVE-1 prompt profile')
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
        import copy
        clip_config = copy.deepcopy(MINECLIP_CONFIG)
        clip_config['ckpt']['path'] = str(self.root / 'weights/steve1/attn.pth')
        self.mineclip = load(clip_config, device=device); self.mineclip.eval()
        prior_config = dict(PRIOR_INFO, model_path=str(self.root / 'weights/steve1/steve1_prior.pt'))
        self.prior = load_vae_model(prior_config); self.embed = get_prior_embed

    def reset(self, goal=None):
        if not goal or not goal.strip(): raise ValueError('STEVE-1 requires a text goal')
        if self.prompt_profile=='official':
            if goal not in self.OFFICIAL_PROMPTS: raise ValueError('unsupported official collection goal')
            prompt=self.OFFICIAL_PROMPTS[goal]
        else: prompt=goal
        self.agent.reset(cond_scale=6.0)
        self.goal_embed = self.embed(prompt, self.mineclip, self.prior, self.device)
        self.condition_metadata={'profile':self.prompt_profile,'requested_goal':goal,'effective_prompt':prompt,
                                 'cond_scale':6.0,'embedding_sha256':hashlib.sha256(self.goal_embed.tobytes()).hexdigest()}

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
    def seed(self, seed): return self.raw.seed(seed)
    def reset(self):
        observation = self.raw.reset(); self.terminal = False; self.initialized = True
        self.last_observation = observation
        return observation
    def step(self, action):
        transition = self.raw.step(action); self.terminal = bool(transition[2])
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
