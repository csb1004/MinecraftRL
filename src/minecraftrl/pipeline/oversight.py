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
from .remedy import choose, outcome, situation, trigger_key
from .steps import Step, item_count

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
# Holds repaired with a remedy (design 2026-11-25-stagnation-remedy, MineEvolve eq. 5): a window without progress, standing
# still, or the same step failing in REPEAT_FAILURES of its last REPEAT_WINDOW attempts.
REMEDY_KINDS = ('stagnation', 'stationary_repetition')
REPEAT_WINDOW, REPEAT_FAILURES = 3, 2
STALL_REPAIR_LIMIT = 30   # 900 s / 30 s windows
EVIDENCE_FOR_INDUCER = ('p', 'move', 'inventory', 'goal', 'window_ticks', 'dominant_action', 'remaining', 'no_progress_ticks',
                        'depth', 'y')


class Oversight:
    def __init__(self, kb, context, goal_item, steps, *, run_id, tools=(), llm=None, selected=None, goal_count=1, skills=None):
        self.kb = kb; self.context = dict(context); self.goal_item = goal_item; self.goal_count = goal_count
        self.run_id = run_id; self.llm = llm; self.selected = selected or {}
        self.guarded, self.needs = oversight(goal_item, steps, tools)
        self.guard = Guard(kb, context, self.guarded); self.monitor = None; self.emit = lambda kind, **fields: None
        self.max_retries = 3; self.retries = Counter()  # per (step id, anomaly kind); config budgets.step_retries
        self.max_rounds = MAX_ROUNDS  # config budgets.knowledge_rounds
        self.skills = dict(skills or {})  # loaded skill specs: switch_skill alternatives share a target item
        self.feedback = []; self.pending = {}  # typed attempt feedback; step id -> remedy claim awaiting its outcome

    def start_step(self, step):
        self.guard.reset(); self.monitor = FastMonitor(step, self.run_id, self.guarded, needs=self.needs)

    def pre_dispatch(self, sample, action): return self.guard.pre_dispatch(sample, action)

    def observe(self, sample):
        """Inline mode: the guard's server-mining check and the step's monitor, both on the control thread."""
        return [e for e in (self.guard.observe(sample), self.monitor.update(sample) if self.monitor else None) if e]

    def observe_guard(self, sample):
        """Thread mode: only the guard runs here (it reads the knowledge base); the monitor runs on its own thread."""
        return self.guard.observe(sample)

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

    def lacking(self, p):
        """What a plan still lacks: the planner's missing subjects, then every block the plan breaks whose harvest
        condition is unknown (the guard would hold at its first attack; check E2, 2026-11-24)."""
        out = list(p.missing)
        for s in p.steps:
            family = block_family(s.block) if s.kind == 'policy' and s.block else None
            if family and family not in out and not self.kb.query(family, 'harvest_requires', context=self.context):
                out.append(family)
        return out

    def plan_ahead(self, initial, inventory=None):
        """Plan the goal with knowledge asked for up front (user request 2026-11-25; MineEvolve plans from retrieved
        knowledge before acting). The first plan used to be whatever the store allowed, naive on an empty store, and
        the knowledge came only when the guard held in the game. Before the game starts, cited web knowledge now fills
        what the plan lacks, by name, for at most `max_rounds` calls. A plan that is still naive or has problems is
        not adopted: the first plan stays and the guard asks at the last moment, as before."""
        inventory = dict(inventory or {}); new = initial
        out = dict(route='plan_ahead', llm='not_needed', reason=None, claim_ids=[], rejected=0, rounds=0, resolution='keep',
                   plan=None, plan_id=None, plan_steps=None, missing=[])
        family = block_family(initial.steps[-1].block) if initial.steps and initial.steps[-1].block else self.goal_item
        for _ in range(self.max_rounds):
            lacking = self.lacking(new)
            if not lacking: break
            if self.llm is None: out.update(llm='not_called', reason=LLM_PENDING); break
            query = dict(goal=self.goal_item, event='goal_received', subject=family, relation='harvest_requires')
            if not new.naive:   # the whole chain was asked for once: now exactly what is still missing
                query.update(missing=lacking, note='return claims only for these subjects; for an item, its recipe or which '
                                                   'block drops it (relation drops with that block family as the subject, '
                                                   'e.g. stone drops cobblestone); for a block family, its harvest_requires '
                                                   '(also when the bare hand is enough)')
            self.emit('llm_request', purpose='plan_ahead', model=self.llm.model, query=query)
            try: raw, meta = acquire(self.llm, query, self.context)
            except (RuntimeError, ValueError) as exc:
                out.update(llm='called' if out['rounds'] else 'failed', reason=str(exc)); break
            accepted, rejected = validate_candidates(raw, meta['citations'])
            ids = store(self.kb, accepted, model=self.llm.model, run_id=self.run_id)
            self.emit('kb_change', claim_ids=ids, rejected=[r['reason'] for r in rejected], citations=meta['citations'], usage=meta['usage'])
            out.update(llm='called', rounds=out['rounds'] + 1, claim_ids=out['claim_ids'] + ids, rejected=out['rejected'] + len(rejected))
            new = plan(self.goal_item, self.goal_count, inventory, self.kb, self.context)
        out['missing'] = self.lacking(new)
        if new is initial or new.naive or new.problems:
            if new is not initial: out['reason'] = 'plan still naive or has problems: %s' % new.problems
            return out
        steps = [Step(**dict(asdict(s), skill=concrete_skill(s.skill, self.selected))) for s in new.steps]
        self.kb.record_plan(new.id, [dict(id=s.id, claim_ids=list(s.claim_ids)) for s in steps])
        self.adopt(new)
        self.emit('plan', plan_id=new.id, steps=[asdict(s) for s in steps], guarded=sorted(new.guarded), needs=sorted(new.needs),
                  mode='replan', reason='plan_ahead')
        return dict(out, resolution='replan', plan=[asdict(s) for s in steps], plan_id=new.id, plan_steps=steps)

    def record_attempt(self, step, feedback):
        """An attempt ended (MineEvolve Monitor): keep its typed feedback and settle a remedy that was waiting on it."""
        index = len(self.feedback); self.feedback.append(feedback); self.settle(step, feedback, index)

    def settle(self, step, feedback, index):
        """The repaired attempt decides the remedy (Curator usage history): completed -> support; stalled or failing the same
        way again -> refutation and dispute. Other endings say nothing about the remedy. Attempt `index` must come after
        the attempt the remedy was chosen for (the held attempt's own feedback is recorded after the slow path)."""
        entry = self.pending.get(step.id)
        if entry is None or index <= entry[1]: return
        claim = self.pending.pop(step.id)[0]
        success = feedback['success']
        if not success and feedback.get('failure_type') not in REMEDY_KINDS + ('guard_violation_persistent',): return
        outcome(self.kb, claim, success, '%s:%s' % (self.run_id, step.id), run_id=self.run_id)
        self.emit('kb_change', claim_ids=[claim], evidence='game_observation_support' if success else 'game_observation_refute',
                  step_id=step.id)

    def repeated(self, step, current):
        recent = [f for f in self.feedback if f['step'] == step.id][-(REPEAT_WINDOW - 1):] + [current]
        return sum(not f['success'] for f in recent) >= REPEAT_FAILURES

    def remedy(self, event, inventory, step, attempt, held):
        """Knowledge, else an approved LLM, else a temporary turn-away; then repair the unfinished plan (MineEvolve
        Adaptor): the completed prefix stays, the held step is tried again after the repair."""
        sit = situation(step, event['kind'], inventory, held, self.skills)
        evidence = {k: v for k, v in (event.get('evidence') or {}).items() if k in EVIDENCE_FOR_INDUCER}
        recent = [f for f in self.feedback if f['step'] == step.id][-(REPEAT_WINDOW - 1):]
        feedback = recent + ([attempt] if attempt else []) + [dict(current_event=event['kind'], evidence=evidence)]
        choice = choose(self.kb, self.llm, self.context, sit, feedback, run_id=self.run_id)
        key = trigger_key(sit['failure_type'], sit['skill']); source, claim, chosen = choice['source'], choice['claim_id'], choice['remedy']
        self.emit('remedy', step_id=step.id, trigger=key, source=source, remedy=chosen, claim_id=claim, rejected=choice['rejected'],
                  usage=choice['usage'])
        if source == 'induced': self.emit('kb_change', claim_ids=[claim], rejected=[], evidence='remedy_induced', step_id=step.id)
        called = choice['usage'] is not None or str(choice['rejected'] or '').startswith('llm_failed')
        out = dict(subject=key, relation='remedy', hits=[claim] if source == 'knowledge' else [], route='remedy',
                   llm='called' if called else 'not_needed' if source == 'knowledge' else 'not_called', reason=choice['rejected'],
                   claim_ids=[claim] if source == 'induced' else [], rejected=int(choice['rejected'] is not None), rounds=int(called),
                   resolution='retry', plan=None, plan_steps=None, remedy=chosen, remedy_source=source, remedy_claim=claim,
                   retry=self.retries[(step.id, event['kind'])])
        action, arg = chosen['action'], chosen.get('arg'); retried = step.id; steps = None; mode = 'insert'
        if action in ('turn_away', 'level_view'): out['prelude'] = action
        elif action == 'switch_skill': out['retry_step'] = Step(**dict(asdict(step), skill=arg))
        elif action == 'equip':
            steps = [Step('equip-' + arg, 'equip', 'rule', arg, 1, params={'goals': [arg], 'equip': arg, 'equip_only': True}), step]
            out['carry'] = True   # the same step again: it keeps the count it started from
        elif action == 'clear_above':
            ceiling = dict(kind='blocked_overhead', evidence=dict(remaining=(event.get('evidence') or {}).get('remaining', step.count)))
            new, _ = self.replan(ceiling, inventory, step)
            if new.problems or not new.steps: out['prelude'] = 'turn_away'; out['reason'] = 'clear_above not plannable: %s' % new.problems
            else: steps = list(new.steps); retried = steps[-1].id
        elif action == 'replan':
            new = plan(self.goal_item, self.goal_count, inventory, self.kb, self.context)
            if new.naive or new.problems: out['prelude'] = 'turn_away'; out['reason'] = 'replan not usable: %s' % new.problems
            else:
                steps = [Step(**dict(asdict(s), skill=concrete_skill(s.skill, self.selected))) for s in new.steps]; mode = 'replan'
                self.adopt(new)
        if steps is not None:
            pid = plan_id(steps)
            self.emit('plan', plan_id=pid, steps=[asdict(s) for s in steps], guarded=sorted(self.guarded), needs=sorted(self.needs),
                      mode=mode, remedy=action)
            out.update(resolution=mode, plan=[asdict(s) for s in steps], plan_id=pid, plan_steps=steps)
        if claim is not None: self.pending[retried] = (claim, len(self.feedback))   # the held attempt's index
        return out

    def adopt(self, new_plan):
        """The RM changed: protect and track what the new plan needs."""
        self.guarded, self.needs = new_plan.guarded, new_plan.needs; self.guard.set_guarded(self.guarded)

    @staticmethod
    def query_for(event):
        kind = event['kind']
        # Aim guards name the block family; the sensor guards 2-4 (drop, wear, hazard) have no knowledge to query yet.
        if kind.startswith('guard_') and 'family' in event: return event['family'], 'harvest_requires'
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

    def slow_path(self, event, inventory, step=None, *, attempt=None, held=None):
        """SQLite first (no LLM on a hit); a gap is filled by cited web knowledge; then the planner decides the new plan.

        A persistent violation is not a knowledge gap (the knowledge already forbids the attack): the policy keeps
        fixating on a protected block, so the step is retried. Every hold at the same step and kind is bounded.
        Stagnation, standing still and a step that keeps failing get a remedy (design 2026-11-25). `attempt` is the
        held attempt's typed feedback, `held` the item in hand."""
        key = (step.id if step else None, event['kind']); self.retries[key] += 1
        if attempt is not None and step is not None: self.settle(step, attempt, len(self.feedback))
        # A stagnation hold needs a full window of game time (30 s), so it is repaired for as many windows as the longest
        # run holds (900 s) instead of the per-step retry limit; every hold stays bounded.
        timed = step is not None and step.kind == 'policy' and event['kind'] in REMEDY_KINDS
        if self.retries[key] > (max(self.max_retries, STALL_REPAIR_LIMIT) if timed else self.max_retries):
            return dict(subject=None, relation=None, hits=[], route='retry_limit', llm='not_needed', claim_ids=[], rejected=0,
                        rounds=0, resolution='stop', plan=None, plan_steps=None,
                        reason='%s held %d times at step %s (limit %d)' % (event['kind'], self.retries[key], key[0], self.max_retries))
        if step is not None and step.kind == 'policy' and (
                event['kind'] in REMEDY_KINDS
                or (event['kind'] == 'guard_violation_persistent' and attempt is not None and self.repeated(step, attempt))):
            return self.remedy(event, inventory, step, attempt, held)
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
