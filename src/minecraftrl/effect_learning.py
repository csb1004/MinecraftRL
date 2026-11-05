"""E1 opt-in effect credit with canonical goals and a frozen official motor."""
from copy import deepcopy
import json

from .conditioned_head import ConditionedHead
from .effect_skill import EffectLedger, EXTRA_FIELDS, effect_context, validate_effect_trace, skill_returns
from .recovery_learning import RecoveryPolicy, RecoveryEnvironment, CONTEXT_FIELDS as RECOVERY_FIELDS
from .rl_policy import state_fingerprint
from .runtime import write_json

SCOPE = 'effect_skill_e1'
CONTEXT_FIELDS = (*RECOVERY_FIELDS, *EXTRA_FIELDS)
VARIANTS = ('nonlinear-on',)
ARCHIVE = 'effect-trajectory.pt'


class EffectHead(ConditionedHead):
    def __init__(self, base, width, variant):
        if variant not in VARIANTS:
            raise ValueError('E1 requires enabled structured conditions')
        super().__init__(base, width, variant, context_dim=len(CONTEXT_FIELDS))
        self.base.requires_grad_(False)


class EffectEnvironment(RecoveryEnvironment):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        p = self.policy
        row = p.learning_rows[-1]
        result = p.effects.commit(row['effect_request'], self.gui, terminal=self.terminal)
        if result['request_id'] != getattr(p, 'last_logged_request', None):
            # Publish only after a real dispatch, never from an unused inference/bootstrap.
            p.write_effect_event('started', row)
            p.last_logged_request = result['request_id']
        row.update(effect_outcome=result, skill_reward=result['skill_reward'], skill_terminal=result['skill_terminal'])
        p.rows[-1].update(effect_request=row['effect_request'], effect_outcome=result)
        if result['skill_terminal']:
            p.write_effect_event('completed' if result['success'] else 'failed', row)
        if p.failure:
            p.failure['effect_request'] = row['effect_request']
            p.failure['effect_outcome'] = result
        return obs, reward, done, info


class EffectPolicy(RecoveryPolicy):
    scope = SCOPE
    context_fields = CONTEXT_FIELDS
    archive_prefix = 'effect'
    environment_type = EffectEnvironment
    max_root_goals = None

    def __init__(self, *args, **kwargs):
        self.effects = EffectLedger()
        super().__init__(*args, **kwargs)
        if self.queue.goal_spec is None:
            raise ValueError('E1 requires a canonical goal_spec_v1')

    def make_actor(self, base, width):
        actor = EffectHead(base, width, self.variant)
        self.official_motor_sha256 = state_fingerprint(actor.base.state_dict())
        return actor

    def semantic_context(self, evidence):
        values, desc = super().semantic_context(evidence)
        self.current_request = self.effects.begin(self.queue)
        values += effect_context(self.current_request, self.effects.last_count)
        desc.update(schema=self.scope, effect_request=self.current_request)
        return values, desc

    def act(self, screen):
        action = super().act(screen)
        self.learning_pending['effect_request'] = deepcopy(self.current_request)
        return action

    def write_effect_event(self, status, row):
        event = dict(schema=self.scope, event=status, step=row['step'], request=row['effect_request'],
            outcome=row.get('effect_outcome'), root_reward=row['reward'], episode_terminal=row['terminal'])
        with (self.output/'skill-events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, sort_keys=True)+'\n')

    def reset(self, goal=None):
        super().reset(goal)
        self.condition_metadata.update(learning_scope=self.scope,
            phase_context_training='canonical root plus committed effect request',
            reward_contract='root reward preserved; skill credit on new inventory high-water mark',
            trainable_scope='residual_adapter_and_value_only')
        write_json(self.output/'rm-condition.json', self.condition_metadata)

    def close(self):
        if self.learning_rows and not self.learning_rows[-1].get('skill_terminal'):
            self.write_effect_event('interrupted_or_truncated', self.learning_rows[-1])
        super().close()

    def archive_learning(self, report):
        super().archive_learning(report)
        report.update(effect_scope=self.scope,
            effect_successes=sum(r.get('skill_reward', 0.) for r in self.learning_rows),
            effect_training_eligible=report.get('s2_training_eligible', False))
