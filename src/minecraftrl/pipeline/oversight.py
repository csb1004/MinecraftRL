"""Goal-derived oversight for one session: guard + per-step monitors + the slow path (SQLite -> LLM -> replan).

Paid LLM calls happen only with an explicitly approved client; without one the slow path stops at the knowledge query.
The game is held (no dispatch) while the slow path runs, so the world does not advance during the call.
"""
import json
import math
from collections import Counter
from dataclasses import asdict

from .guard import Guard
from .kb import RELATIONS, Evidence
from .knowledge import acquire, hypothesize, store, validate_candidates, validate_hypothesis
from .monitor import FastMonitor
from .planner import Plan, block_family, clear_plan, concrete_skill, escape_plan, harvest_verdict, oversight, plan, plan_id
from .steps import item_count

LLM_PENDING = 'paid API calls need explicit approval (--allow-paid); not called'
# Knowledge acquisition calls per hold (the run budget still caps the total). Round 3b (2026-11-05): a chain can lack its
# links one at a time (stone pickaxe -> wooden pickaxe -> planks/sticks), so two rounds were not enough.
MAX_ROUNDS = 4
# The craft skill's own precondition failure (workshop.place_table); below the surface it means the 1-wide pit.
PIT_REASON = 'no reachable safe placement surface'
PLANNED = ('harvest_requires', 'escape_by', 'blocks_movement')  # relations whose knowledge the planner turns into a plan
# Capabilities a disclosed rule skill stands in for (spec 5.10: V1 records learning requests, does not train).
LEARNING_REQUESTS = {'clear_above': dict(capability='clear the blocks over the head while climbing',
                                         policy_candidates=['mine the ceiling', 'break the block above you', 'dig up'],
                                         evidence='results/pit-ceiling-probe-20261104 (0/9; mine the ceiling aims up 97-100% but attacks 1-5%)')}


class Oversight:
    def __init__(self, kb, context, goal_item, steps, *, run_id, tools=(), llm=None, selected=None, goal_count=1):
        self.kb = kb; self.context = dict(context); self.goal_item = goal_item; self.goal_count = goal_count
        self.run_id = run_id; self.llm = llm; self.selected = selected or {}
        self.guarded, self.needs = oversight(goal_item, steps, tools)
        self.guard = Guard(kb, context, self.guarded); self.monitor = None; self.emit = lambda kind, **fields: None
        self.max_retries = 3; self.retries = Counter()  # per (step id, anomaly kind); config budgets.step_retries
        self.max_rounds = MAX_ROUNDS  # config budgets.knowledge_rounds

    def start_step(self, step):
        self.guard.reset(); self.monitor = FastMonitor(step, self.run_id, self.guarded, needs=self.needs)

    def pre_dispatch(self, sample, action): return self.guard.pre_dispatch(sample, action)

    def observe(self, sample):
        return [e for e in (self.guard.observe(sample), self.monitor.update(sample) if self.monitor else None) if e]

    def rm_state(self, step):
        return dict(goal=self.goal_item, guarded=sorted(self.guarded), needs=sorted(self.needs), active_step=step.id,
                    expect='rise' if step.params.get('expect') == 'rise' else 'gain:' + step.target)

    def summary(self):
        calls = self.llm.calls if self.llm else []
        return dict(goal=self.goal_item, guarded=sorted(self.guarded), needs=sorted(self.needs), llm_model=getattr(self.llm, 'model', None),
                    api_calls=len(calls), usage=[c.get('usage') for c in calls],
                    call_log=[{k: v for k, v in c.items() if k != 'usage'} for c in calls])  # ok / error type / HTTP status

    def refute_permission(self, event, step):
        """A guarded block broke without a drop: the knowledge that let the guard allow it gets a game refutation and goes
        through conflict handling (spec 2 post-hoc scene, plan Task 8). Returns the observation for the next question."""
        evidence = event.get('evidence') or {}; block = evidence.get('block'); held = evidence.get('held') or 'air'
        observation = dict(broke=block, held=held, harvestable=False, note='the block broke and dropped nothing')
        ref = '%s:%s' % (self.run_id, step.id if step else None)
        for claim in self.kb.query(block_family(block), 'harvest_requires', context=self.context):
            if harvest_verdict([claim], held)[0] != 'ok': continue
            outcome = self.kb.resolve_conflict(claim.id, Evidence('game_observation', 'refute', ref, json.dumps(observation)), self.context,
                                               candidate_condition=None, observation_valid=True, run_id=self.run_id)
            self.emit('knowledge_conflict', claim_id=claim.id, subject=claim.subject, relation='harvest_requires',
                      expected=claim.object, observed=observation, outcome=outcome, hypothesis=None)
        return observation

    def verify(self, step, gained, sample):
        """A completed step (server-judged): the knowledge it used gains game support (spec 5.8); a drops claim the
        game contradicts goes to conflict handling (spec 5.5): observation error -> cited condition split -> dispute."""
        ref = '%s:%s' % (self.run_id, step.id)
        for claim_id in step.claim_ids:
            self.kb.add_evidence(claim_id, Evidence('game_observation', 'support', ref, 'step completed'), run_id=self.run_id)
        if step.claim_ids: self.emit('kb_change', claim_ids=list(step.claim_ids), evidence='game_observation_support', step_id=step.id)
        if step.kind != 'policy' or not step.block or not gained: return
        family = block_family(step.block)
        valid = sample is not None and block_family(sample.last_destroyed) == family and sample.last_harvestable
        for claim in self.kb.query(family, 'drops', context=self.context):
            if item_count(gained, claim.object) > 0: continue
            observation = dict(broke=sample.last_destroyed if sample else None, held=sample.selected_item if sample else None,
                               gained=dict(gained))
            hypothesis = None
            if self.llm is not None and valid:
                self.emit('llm_request', purpose='condition_hypothesis', model=self.llm.model, claim_id=claim.id)
                try:
                    raw, meta = hypothesize(self.llm, dict(subject=claim.subject, relation=claim.relation, object=claim.object,
                                                           conditions=claim.conditions), observation, self.context)
                    hypothesis = dict(validate_hypothesis(raw, meta['citations']), usage=meta['usage'])
                except (RuntimeError, ValueError) as exc: hypothesis = dict(condition=None, reason='llm_failed: ' + str(exc))
            outcome = self.kb.resolve_conflict(
                claim.id, Evidence('game_observation', 'refute', ref, json.dumps(observation)), self.context,
                candidate_condition=(hypothesis or {}).get('condition'), observation_valid=valid, run_id=self.run_id)
            self.emit('knowledge_conflict', claim_id=claim.id, subject=claim.subject, relation='drops', expected=claim.object,
                      observed=dict(gained), outcome=outcome, hypothesis=hypothesis)

    def adopt(self, new_plan):
        """The RM changed: protect and track what the new plan needs."""
        self.guarded, self.needs = new_plan.guarded, new_plan.needs; self.guard.set_guarded(self.guarded)

    @staticmethod
    def query_for(event):
        kind = event['kind']
        if kind.startswith('guard_'): return event['family'], 'harvest_requires'
        if kind == 'destroy_not_harvestable': return block_family(event['evidence']['block']), 'harvest_requires'
        if kind == 'blocked_overhead': return 'solid_block', 'blocks_movement'
        deep = (event.get('evidence', {}).get('depth') or 0) >= 2
        if kind == 'rule_skill_failed' and PIT_REASON in event['evidence'].get('reason', '') and deep:
            return 'vertical_pit', 'escape_by'  # spec 5.3: the blocked next transition adds the escape sub-goal
        if kind == 'climb_without_block' and deep: return 'vertical_pit', 'escape_by'  # replan from the current height
        return None, None

    def replan(self, event, inventory, step=None):
        """(plan, mode): 'replan' replaces the queue; 'insert' puts a sub-goal in front of the remaining queue."""
        relation = self.query_for(event)[1]
        if relation == 'blocks_movement':
            steps, problems, missing = clear_plan(self.kb, self.context)
            if step is not None and steps:  # then the same climb for the height still to go
                remaining = max(1, math.ceil(event['evidence'].get('remaining', step.count) - 1e-6))
                # One continuation id, so repeated ceiling holds share the per-step retry limit (real check C1).
                after = step.id if step.id.endswith('-after-clear') else step.id + '-after-clear'
                steps = steps + [type(step)(**dict(asdict(step), id=after, count=remaining))]
            return Plan(plan_id(steps), steps, False, problems, self.guarded, self.needs, missing), 'insert'
        if relation == 'escape_by':
            return escape_plan(self.goal_item, self.goal_count, inventory, event['evidence']['depth'], self.kb, self.context), 'replan'
        return plan(self.goal_item, self.goal_count, inventory, self.kb, self.context), 'replan'

    def slow_path(self, event, inventory, step=None):
        """SQLite first (no LLM on a hit); a gap is filled by cited web knowledge; then the planner decides the new plan.

        A persistent violation is not a knowledge gap (the knowledge already forbids the attack): the policy keeps
        fixating on a protected block, so the step is retried. Every hold at the same step and kind is bounded."""
        key = (step.id if step else None, event['kind']); self.retries[key] += 1
        if self.retries[key] > self.max_retries:
            return dict(subject=None, relation=None, hits=[], route='retry_limit', llm='not_needed', claim_ids=[], rejected=0,
                        rounds=0, resolution='stop', plan=None, plan_steps=None,
                        reason='%s held %d times at step %s (limit %d)' % (event['kind'], self.retries[key], key[0], self.max_retries))
        if event['kind'] == 'guard_violation_persistent':
            return dict(subject=event['family'], relation='harvest_requires', hits=list(event['claim_ids']), route='retry_stage',
                        llm='not_needed', reason=None, claim_ids=[], rejected=0, rounds=0, resolution='retry', retry=self.retries[key],
                        plan=None, plan_steps=None)
        subject, relation = self.query_for(event)
        observation = self.refute_permission(event, step) if event['kind'] == 'destroy_not_harvestable' else None
        if event['kind'] == 'rule_skill_failed' and relation is None:
            # Spec 5.3 default when no decision exists (the LLM decision step is not built): retry the same step, bounded.
            return dict(subject=None, relation=None, hits=[], route='retry_stage', llm='not_called',
                        reason='no decision prompt for this failure yet; default retry', claim_ids=[], rejected=0, rounds=0,
                        resolution='retry', retry=self.retries[key], plan=None, plan_steps=None)
        hits = [c.id for c in self.kb.query(subject, relation, context=self.context)] if relation in RELATIONS else []
        route = 'ask_llm' if subject is None else 'replan_from_kb' if hits else 'acquire_then_replan'
        out = dict(subject=subject, relation=relation, hits=hits, route=route, llm='not_needed', reason=None,
                   claim_ids=[], rejected=0, rounds=0, resolution='stop', plan=None, plan_steps=None)
        if route == 'ask_llm': return dict(out, llm='not_called', reason='no decision prompt for this event kind yet')
        query = dict(goal=self.goal_item, event=event['kind'], subject=subject, relation=relation)
        if observation: query['observation'] = observation  # what the game showed against the stored knowledge
        if relation == 'escape_by':  # what the LLM needs to answer for this pit (server observation, disclosed)
            query['situation'] = dict(depth=event['evidence']['depth'], width=1, reason=event['evidence'].get('reason', event['kind']),
                                      inventory=inventory)
        new = None
        for _ in range(self.max_rounds):
            if route == 'replan_from_kb' and out['rounds'] == 0 and new is None: pass  # SQLite hit: plan before asking
            elif self.llm is None: return dict(out, llm='not_called', reason=LLM_PENDING)
            else:
                self.emit('llm_request', purpose='acquire_knowledge', model=self.llm.model, query=query)
                try: raw, meta = acquire(self.llm, query, self.context)
                except (RuntimeError, ValueError) as exc:
                    # A failure after earlier useful rounds (e.g. the budget ran out) is a knowledge shortfall, not an API failure.
                    return dict(out, llm='called' if out['rounds'] else 'failed', reason=str(exc))
                accepted, rejected = validate_candidates(raw, meta['citations'])
                ids = store(self.kb, accepted, model=self.llm.model, run_id=self.run_id)
                self.emit('kb_change', claim_ids=ids, rejected=[r['reason'] for r in rejected], citations=meta['citations'], usage=meta['usage'])
                out.update(llm='called', rounds=out['rounds'] + 1, claim_ids=out['claim_ids'] + ids,
                           rejected=out['rejected'] + len(rejected), citations=meta['citations'])
            if relation not in PLANNED: return dict(out, reason='knowledge stored; no executable plan for this event kind yet')
            new, mode = self.replan(event, inventory, step)
            if not new.naive and not new.problems: break
            if not new.missing: break  # a problem knowledge cannot fix (e.g. too few blocks): asking again is waste
            # Ask once more for exactly what the plan still lacks (no knowledge is not "no condition").
            query = dict(query, missing=list(new.missing),
                         note='return claims only for these subjects; for an item, its recipe or which block drops it '
                              '(relation drops with that block family as the subject, e.g. stone drops cobblestone)')
        if new is None or new.naive or new.problems:
            return dict(out, reason='replan still naive or has problems: %s' % (new.problems if new else None))
        steps = [type(s)(**dict(asdict(s), skill=concrete_skill(s.skill, self.selected))) for s in new.steps]
        self.kb.record_plan(new.id, [dict(id=s.id, claim_ids=list(s.claim_ids)) for s in steps])
        self.adopt(new)
        self.emit('plan', plan_id=new.id, steps=[asdict(s) for s in steps], guarded=sorted(new.guarded), needs=sorted(new.needs), mode=mode)
        for s in steps:
            if s.skill in LEARNING_REQUESTS:
                self.emit('learning_request', rule_skill=s.skill, step_id=s.id, status='recorded_not_executed', **LEARNING_REQUESTS[s.skill])
        return dict(out, resolution=mode, plan=[asdict(s) for s in steps], plan_id=new.id, plan_steps=steps)
