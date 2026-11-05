"""Frozen official STEVE-1 RM goal-conditioning pilot, not an RL learner."""
import hashlib
import json
from pathlib import Path
import time

from .policies import Steve1Policy
from .rl_collect import RLEnvironment
from .rl_core import LOGS, ShapedWoodRewardV3
from .runtime import jsonable, run_episode, write_json

PROMPTS = {
    'root': 'collect wood',
    'search': 'go explore',
    'approach': 'approach a tree',
    'mine': 'chop down the tree',
    'collect': 'pick up wood',
}
STAGES = ('search', 'approach', 'mine', 'collect', 'complete')


class StageSelector:
    """Debounce nonterminal telemetry; accepting states always stop immediately."""
    def __init__(self, confirmations=3):
        if type(confirmations) is not int or confirmations < 1:
            raise ValueError('positive integer confirmation count required')
        self.confirmations = confirmations
        self.current = self.candidate = None
        self.count = 0

    def observe(self, stage):
        if stage not in STAGES:
            raise ValueError('unknown RM stage')
        if self.current == 'complete' and stage != 'complete':
            raise RuntimeError('RM cannot resume after completion')
        if self.current is None or stage == 'complete':
            self.current = stage
        self.count = self.count + 1 if stage == self.candidate else 1
        self.candidate = stage
        if self.count >= self.confirmations:
            self.current = stage
        return self.current


def load_bank(path):
    import numpy as np
    path = Path(path).resolve()
    meta = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('schema') != 1 or meta.get('prompts') != PROMPTS or meta.get('seed') != 31031:
        raise ValueError('goal bank does not match P1 protocol')
    if set(meta.get('embeddings', {})) != set(PROMPTS):
        raise ValueError('missing goal embedding')
    embeddings = {}
    for key in PROMPTS:
        raw = (path/(key+'.npy')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['embeddings'][key]['file_sha256']:
            raise ValueError('goal bank hash mismatch: '+key)
        import io
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        if value.shape != (1, 512) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('invalid goal embedding: '+key)
        if hashlib.sha256(value.tobytes()).hexdigest() != meta['embeddings'][key]['vector_sha256']:
            raise ValueError('goal vector hash mismatch: '+key)
        embeddings[key] = value
    return embeddings, meta


def validate_args(args):
    s2_variant=getattr(args,'s2_variant',None)
    s2_checkpoint=getattr(args,'s2_checkpoint',None)
    if s2_variant or s2_checkpoint:
        if (s2_variant not in ('linear-off','linear-on','nonlinear-off','nonlinear-on')
                or getattr(args,'rm_conditioning',None)!='goal-queue'
                or getattr(args,'s1_record',False) or getattr(args,'s1_checkpoint',None)):
            raise ValueError('S2 requires an explicit isolated goal-queue variant')
        if not 1 <= getattr(args,'s2_action_budget',1200) <= 1200:
            raise ValueError('S2 action budget must be 1 to 1200')
        if s2_checkpoint is not None and (not Path(s2_checkpoint).resolve().is_relative_to(args.root.resolve()/'results')
                                          or not Path(s2_checkpoint).is_file()):
            raise ValueError('existing project-local S2 checkpoint required')
    mode = getattr(args, 'rm_conditioning', None)
    bank = getattr(args, 'rm_goal_bank', None)
    program = getattr(args, 'rm_program', None)
    learning = getattr(args, 's1_record', False)
    checkpoint = getattr(args, 's1_checkpoint', None)
    if learning and mode != 'goal-queue':
        raise ValueError('S1 recording requires the priority goal program')
    if (checkpoint is not None or getattr(args,'s1_split','development')!='development') and not learning:
        raise ValueError('S1 checkpoint/split requires explicit recording mode')
    if checkpoint is not None and (not Path(checkpoint).resolve().is_relative_to(args.root.resolve()/'results')
                                   or not Path(checkpoint).is_file()):
        raise ValueError('existing project-local S1 checkpoint required')
    if program is not None and mode != 'goal-queue':
        raise ValueError('goal program requires goal-queue mode')
    if mode is None and bank is None:
        return
    if (mode not in ('fixed', 'staged', 'official', 'purpose-fixed', 'purpose-switch', 'logo-staged', 'logo-fixed', 'goal-queue') or bank is None or args.policy != 'steve1'
            or args.condition != 'wood' or args.operation not in ('episode', '_worker')
            or any(getattr(args, name, False) for name in
                   ('rl_collect', 'task_session', 'world_slot', 'craft_probe', 'craft_workshop',
                    'craft_recipe_book', 'tool_collection', 'task_recovery_catalogue', 'task_policy_memory'))
            or getattr(args, 'steve_cond_scale', None) is not None
            or getattr(args, 'steve_prompt_profile', 'short') != 'short'):
        raise ValueError('RM conditioning requires isolated official STEVE-1 wood episode and goal bank')
    if not Path(bank).resolve().is_relative_to(args.root.resolve()/'results'):
        raise ValueError('goal bank must be inside project results')
    if mode == 'goal-queue':
        from .goal_spec import resolve_program
        from .goal_contract import read_json
        if program is None or not Path(program).resolve().is_relative_to(args.root.resolve()):
            raise ValueError('project-local goal program required')
        _, compiled = resolve_program(read_json(program))
        if compiled is not None:
            variant = getattr(args,'s2_variant',None) or getattr(args,'recovery_variant',None)
            if variant not in ('linear-on','nonlinear-on'):
                raise ValueError('goal_spec_v1 requires enabled S2/R1 structured policy context')
            if compiled['source']['schema']=='goal_spec_v2' and not getattr(args,'execution_learning',False):
                raise ValueError('goal_spec_v2 requires E2 execution')
            if (getattr(args,'recovery_variant',None) and not getattr(args,'effect_learning',False)
                    and not getattr(args,'execution_learning',False)
                    and len(compiled['program']['goals']) != 1):
                raise ValueError('R1 recovery requires exactly one goal')


class ConditioningPolicy(Steve1Policy):
    bank_loader = staticmethod(load_bank)

    def __init__(self, env, root, output, mode, bank, seed, is_stopped):
        self.embeddings, self.bank_metadata = self.bank_loader(bank)
        self.prompts = self.bank_metadata['prompts']
        for name in ('attn.pth', 'steve1_prior.pt', 'steve1.weights', '2x.model'):
            receipt = json.loads((Path(root)/'weights/steve1'/(name+'.receipt.json')).read_text())
            if self.bank_metadata.get('official_weights', {}).get(name) != receipt['sha256']:
                raise ValueError('goal bank official model mismatch: '+name)
        # Load only the official action policy; the separately prepared goal bank
        # avoids co-resident MineCLIP memory. reset below never loads visual prompts.
        super().__init__(env, root, prompt_profile='official-visual', guidance=6.)
        self.agent.policy.requires_grad_(False)
        self.output, self.mode, self.seed = Path(output), mode, seed
        self.is_stopped = is_stopped
        self.selector = StageSelector()
        self.raw_stage = None
        self.rows, self.switches = [], []
        self.pending = self.pending_state = None
        self.last_key = None
        self.controller_name = 'official_steve1_rm_'+mode
        from .cuda_startup import prime_convolution
        write_json(self.output/'backend-preparation.json',
                   prime_convolution(self.agent.policy.net.img_process.cnn.stacks[0].firstconv.layer))

    def observe_stage(self, stage):
        self.raw_stage = stage
        self.selector.observe(stage)

    def goal_key(self):
        return 'root' if self.mode == 'fixed' else self.selector.current

    def row_context(self):
        return {}

    def reset(self, goal=None):
        if goal != 'collect wood' or self.selector.current is None:
            raise ValueError('RM initialized before wood policy reset required')
        self.agent.reset(cond_scale=6.)
        self.goal_embed = self.embeddings['root'].copy()
        self.torch.manual_seed(self.seed)
        self.torch.cuda.manual_seed_all(self.seed)
        self.condition_metadata = dict(
            controller=self.controller_name, weights='official_frozen', optimizer_updates=0,
            mode=self.mode, goal='collect wood', guidance=6., sampling_seed=self.seed,
            stage_source='privileged_server_wood_v3', memory_on_switch='preserve',
            stage_confirmations=3, bank=self.bank_metadata, recurrent_commit='dispatch_only')
        write_json(self.output/'rm-condition.json', self.condition_metadata)

    def act(self, screen):
        if self.is_stopped() or self.selector.current == 'complete':
            raise InterruptedError('STOP/complete before RM policy inference')
        if self.pending is not None or self.selector.current is None:
            raise RuntimeError('missing stage or uncommitted policy action')
        key = self.goal_key()
        self.goal_embed = self.embeddings[key].copy()
        previous_state = self.agent.hidden_state
        try:
            action = super().act(screen)
            self.pending_state = self.agent.hidden_state
        finally:
            self.agent.hidden_state = previous_state
        self.pending = dict(stage=self.selector.current, raw_stage=self.raw_stage,
                            goal_key=key, prompt=self.prompts[key],
                            embedding_sha256=self.bank_metadata['embeddings'][key]['vector_sha256'],
                            env_action=jsonable(action))
        self.pending.update(self.row_context())
        return action

    def commit(self, action, reward, terminal):
        if self.pending is None or jsonable(action) != self.pending['env_action']:
            raise ValueError('RM action/dispatch mismatch')
        row = dict(self.pending, reward=reward, terminal=terminal, step=len(self.rows)+1)
        self.rows.append(row)
        if row['goal_key'] != self.last_key:
            self.switches.append(dict(step=row['step'], previous=self.last_key,
                                      goal_key=row['goal_key'], stage=row['stage'],
                                      embedding_sha256=row['embedding_sha256']))
        self.last_key = row['goal_key']
        self.agent.hidden_state = self.pending_state
        self.pending = self.pending_state = None

    def close(self):
        # Release tensors inside the control cleanup, retaining small audit rows.
        self.pending_state = self.pending = None
        super().close()


class ConditioningEnvironment(RLEnvironment):
    def update_stage(self):
        stage = 'complete' if self.reward.completed else ShapedWoodRewardV3.state(self.gui)[0]
        self.policy.observe_stage(stage)

    def authoritative(self, observation):
        return dict(observation, inventory=dict(self.gui['server_inventory']))

    def reset(self):
        obs = super().reset()
        self.update_stage()
        return self.authoritative(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.update_stage()
        return self.authoritative(obs), reward, done, info


def run_conditioning(policy, env, output, **kwargs):
    def stopped():
        return (kwargs['gate'].requested.is_set() or
                bool(kwargs.get('stop_file') and Path(kwargs['stop_file']).exists()))
    environment_type = getattr(policy, 'environment_type', ConditioningEnvironment)
    wrapper = environment_type(env, policy, strict_ticks=True, is_stopped=stopped)
    def control_closed(report):
        # Called only after latched STOP, verified release and policy closure.
        # Slow game/interpreter teardown gets the existing bounded archive grace;
        # cancellation remains invalid for performance and training.
        write_json(Path(output)/'rm-control-finished.json', dict(
            control_closed=True, release=report['release'],
            cancelled_archive=report['status']=='cancelled', monotonic_s=time.monotonic(),
            scope='input release and policy closure verified; no further game actions'))
    runtime_goal=('Complete goal queue: '+', '.join(policy.queue.goals) if hasattr(policy,'queue') else 'collect wood')
    report = run_episode(policy, wrapper, output, goal=runtime_goal,
                         target_items=() if hasattr(policy,'queue') else LOGS,
                         completion_check=lambda obs: wrapper.reward.completed,
                         check_completion_on_reset=getattr(policy,'execution_contracts',False),
                         on_control_closed=control_closed, **kwargs)
    write_json(Path(output)/'rm-rows.json', policy.rows)
    write_json(Path(output)/'rm-switches.json', policy.switches)
    write_json(Path(output)/'initial-state.json', getattr(wrapper, 'initial', {}))
    report.update(rm_conditioning=policy.mode, optimizer_updates=0, learned_policy_improvement=False,
                  server_target_reached=bool(wrapper.reward and wrapper.reward.completed),
                  condition=policy.condition_metadata, stage_counts={stage:sum(row['stage']==stage for row in policy.rows)
                                                                    for stage in STAGES},
                  goal_switch_count=max(0, len(policy.switches)-1),
                  server_tick_deltas=wrapper.ticks, simulation_ticks=sum(wrapper.ticks),
                  information_condition='privileged_server_RM', training_eligible=False)
    if hasattr(policy,'queue'):
        report.update(goal_queue=policy.queue.snapshot(),
            completed_goal_count=sum(r['status']=='completed' for r in policy.queue.records.values()),
            program_completed=policy.queue.completed,target_observed=policy.queue.completed,
            last_skill_progress=policy.phase_snapshot)
    if hasattr(policy,'archive_learning'):
        policy.archive_learning(report)
    write_json(Path(output)/'summary.json', report)
    return report
