"""LOGO-inspired generated validators over a caller-owned, fixed observer.

This is a bounded expression compiler, not an executor for LLM-written Python.
The caller owns facts, the root success condition, and acceptance examples.
"""
from copy import deepcopy
import math
from numbers import Real
import operator

from .goal_contract import digest, fields, identifier, object_schema

OPS = {'eq': operator.eq, 'ge': operator.ge, 'le': operator.le}
INSTRUCTIONS = '''Return validator expressions using only the response schema and supplied facts.
Define subgoal success as an observed state change, never an attempted action or a language claim.
Use delta for acquisition/break counters; existing inventory/counters are not new achievement.
Preserve the caller's root requirement exactly. A known target is not necessarily currently visible.
Missing or stale observations yield unknown, never success. Do not invent an observer or execute code.
Return unsupported when the fixed facts cannot establish a requested result.
Candidates must pass caller-owned positive, negative and missing-evidence checks before admission.
'''


def number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and abs(value) <= 1e12 and math.isfinite(value)


def validate_expression(expr, facts, depth=0, budget=None):
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > 6 or budget[0] > 64 or not isinstance(expr, dict):
        raise ValueError('invalid or oversized validator expression')
    if expr.get('op') in ('all', 'any'):
        fields(expr, 'op args', 'boolean expression')
        if not isinstance(expr['args'], list) or not 1 <= len(expr['args']) <= 8:
            raise ValueError('bounded nonempty operands required')
        for child in expr['args']:
            validate_expression(child, facts, depth + 1, budget)
    else:
        fields(expr, 'op fact mode value', 'comparison')
        if (not isinstance(expr['op'], str) or not isinstance(expr['fact'], str)
                or expr['op'] not in OPS or expr['fact'] not in facts
                or expr['mode'] not in ('current', 'delta') or not number(expr['value'])):
            raise ValueError('unsupported fact, comparison or threshold')


def validate_context(context):
    fields(context, 'schema task facts validators root root_requirement', 'validator context')
    if context['schema'] != 'logo_context_v1' or not isinstance(context['task'], str) or not context['task']:
        raise ValueError('invalid validator context')
    facts = context['facts']
    if not isinstance(facts, dict) or not 1 <= len(facts) <= 64:
        raise ValueError('bounded fact catalog required')
    for ref, fact in facts.items():
        identifier(ref)
        fields(fact, 'path items description', 'fact')
        if (not isinstance(fact['path'], list) or not 1 <= len(fact['path']) <= 4
                or any(not isinstance(key, str) or not key for key in fact['path'])
                or not isinstance(fact['items'], list) or len(fact['items']) > 64
                or any(not isinstance(key, str) or not key for key in fact['items'])
                or len(fact['items']) != len(set(fact['items']))
                or not isinstance(fact['description'], str) or not fact['description']):
            raise ValueError('invalid fixed fact descriptor')
    names = context['validators']
    if not isinstance(names, list) or not 1 <= len(names) <= 32 or len(names) != len(set(names)):
        raise ValueError('invalid required validators')
    for name in names:
        identifier(name)
    if context['root'] not in names:
        raise ValueError('missing root validator')
    validate_expression(context['root_requirement'], facts)


def request(context):
    validate_context(context)
    expression = dict(anyOf=[
        object_schema(dict(op=dict(type='string', enum=['all', 'any']),
                           args=dict(type='array', minItems=1, maxItems=8, items={'$ref': '#/$defs/expression'}))),
        object_schema(dict(op=dict(type='string', enum=list(OPS)),
                           fact=dict(type='string', enum=list(context['facts'])),
                           mode=dict(type='string', enum=['current', 'delta']),
                           value=dict(type='number', minimum=-1e12, maximum=1e12)))])
    schema = object_schema(dict(schema=dict(type='string', enum=['logo_proposal_v1']),
        context_sha256=dict(type='string', enum=[digest(context)]),
        status=dict(type='string', enum=['candidate', 'unsupported']), reason=dict(type='string'),
        validators=dict(type='array', maxItems=32, items=object_schema(dict(
            id=dict(type='string', enum=context['validators']), expression={'$ref': '#/$defs/expression'})))))
    schema['$defs'] = dict(expression=expression)
    return dict(instructions=INSTRUCTIONS, context=deepcopy(context), context_sha256=digest(context), response_schema=schema)


def fact_value(descriptor, observation):
    value = observation
    for key in descriptor['path']:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if descriptor['items']:
        if not isinstance(value, dict):
            return None
        counts = [value.get(key) for key in descriptor['items']]
        if any(not number(count) or count < 0 or int(count) != count for count in counts):
            return None
        value = sum(counts)
    return value if number(value) else None


def observation_valid(observation):
    return (isinstance(observation, dict) and number(observation.get('server_available'))
            and observation['server_available'] == 1 and number(observation.get('error'))
            and observation['error'] == 0 and number(observation.get('server_tick'))
            and observation['server_tick'] >= 0 and int(observation['server_tick']) == observation['server_tick'])


def evaluate(expr, current, baseline, facts):
    if expr['op'] in ('all', 'any'):
        values = [evaluate(child, current, baseline, facts) for child in expr['args']]
        # Conservative: incomplete evidence cannot establish either branch.
        if None in values:
            return None
        return all(values) if expr['op'] == 'all' else any(values)
    value = fact_value(facts[expr['fact']], current)
    start = fact_value(facts[expr['fact']], baseline) if expr['mode'] == 'delta' else 0
    if value is None or start is None:
        return None
    return bool(OPS[expr['op']](value - start, expr['value']))


class ValidatorSet:
    """Compiled callable functions; None means unavailable evidence, not False."""
    def __init__(self, context, proposal):
        validate_context(context)
        fields(proposal, 'schema context_sha256 status reason validators', 'validator proposal')
        if proposal['schema'] != 'logo_proposal_v1' or proposal['context_sha256'] != digest(context):
            raise ValueError('stale or invalid validator proposal')
        if proposal['status'] != 'candidate':
            raise ValueError('unsupported validator proposal: ' + str(proposal['reason']))
        if not isinstance(proposal['reason'], str) or len(proposal['reason']) > 1000:
            raise ValueError('invalid rationale')
        if not isinstance(proposal['validators'], list) or len(proposal['validators']) > 32:
            raise ValueError('invalid validator list')
        expressions = {}
        for item in proposal['validators']:
            fields(item, 'id expression', 'validator')
            identifier(item['id'])
            if item['id'] not in context['validators'] or item['id'] in expressions:
                raise ValueError('unknown or duplicate validator')
            validate_expression(item['expression'], context['facts'])
            expressions[item['id']] = deepcopy(item['expression'])
        if set(expressions) != set(context['validators']):
            raise ValueError('missing required validator')
        if expressions[context['root']] != context['root_requirement']:
            raise ValueError('generated root cannot weaken or replace the caller success criterion')
        self.context, self.proposal = deepcopy(context), deepcopy(proposal)
        self.expressions = expressions
        self.sha256 = digest(dict(context=context, proposal=proposal))
        self.functions = {name: self._function(expr) for name, expr in expressions.items()}

    def _function(self, expression):
        def predicate(current, baseline):
            if (not observation_valid(current) or not observation_valid(baseline)
                    or current['server_tick'] < baseline['server_tick']):
                return None
            return evaluate(expression, current, baseline, self.context['facts'])
        return predicate

    def observe(self, current, baseline):
        return {name: fn(current, baseline) for name, fn in self.functions.items()}

    def verify(self, cases):
        """Cases must come from the caller, never from the generated proposal."""
        failures, coverage = [], {name: set() for name in self.functions}
        for case in cases:
            actual = self.observe(case['current'], case['baseline'])
            for name, expected in case['expected'].items():
                if name not in actual or (expected is not None and type(expected) is not bool):
                    raise ValueError('invalid acceptance example')
                coverage[name].add(expected)
                if actual[name] is not expected:
                    failures.append(dict(case=case['name'], validator=name, expected=expected, actual=actual[name]))
        missing = [name for name, values in coverage.items() if not {True, False}.issubset(values)]
        return dict(schema='logo_review_v1', validator_sha256=self.sha256, cases_sha256=digest(cases),
                    passed=not failures and not missing, failures=failures, missing_coverage=missing,
                    scope='deterministic contract examples; not a proof of semantic completeness or policy capability')


class SuccessMonitor:
    """One sparse success reward, strictly advancing server observations, absorbing completion."""
    def __init__(self, validators, baseline):
        self.validators, self.baseline = validators, deepcopy(baseline)
        self.last_tick = baseline.get('server_tick')
        self.values = validators.observe(baseline, baseline)
        if any(value is None for value in self.values.values()):
            raise ValueError('unknown baseline validator evidence')
        if self.values[validators.context['root']]:
            raise ValueError('root already satisfied at baseline')
        self.completed = False

    def advance(self, current):
        if self.completed:
            raise RuntimeError('observation after accepting state')
        if not observation_valid(current) or current['server_tick'] <= self.last_tick:
            raise ValueError('missing or stale server observation')
        values = self.validators.observe(current, self.baseline)
        completed = values[self.validators.context['root']] is True
        if not completed and any(value is None for value in values.values()):
            raise ValueError('unknown validator evidence; execution cannot continue')
        self.values, self.last_tick = values, current['server_tick']
        self.completed = completed
        return float(self.completed)
