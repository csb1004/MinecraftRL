"""Goal-specific monitor specs (plan docs/specs/2026-11-05-goal-monitor-spec-plan.md §4, §5, §8): schema and checks.

A spec says, per sub-goal, how it ends (bounded or until_found) and which fixed signals to watch, plus the constraints
a person wrote. It never carries time numbers (they come from records) or success conditions (the caller owns
success), and signals come from the fixed list only (no exec/eval). Every author gets the same checks; `author` is
recorded only. Constraints from an LLM are rejected (user decision 2026-11-05).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

ENDINGS = ('bounded', 'until_found')
UNITS = ('step', 'item', 'block')
SIGNALS = {
    'objective_inventory_increased': ('progress', ENDINGS),
    'new_position_cell': ('progress', ('until_found',)),
    'new_search_view': ('progress', ('until_found',)),
    'rise': ('progress', ('bounded',)),
    'no_progress': ('anomaly', ('until_found',)),
    'repeated_cycle': ('anomaly', ('until_found',)),
    'stationary_repetition': ('anomaly', ENDINGS),
    'harvest_without_drop': ('anomaly', ENDINGS),
    'needed_tool_lost': ('anomaly', ENDINGS),
    'low_health': ('anomaly', ENDINGS),
    'blocked_overhead': ('anomaly', ('bounded',)),
    'climb_without_block': ('anomaly', ('bounded',)),
    'moving_but_stuck': ('waste', ENDINGS),
    'looking_down': ('waste', ENDINGS),
    'revisiting_area': ('waste', ('until_found',)),
}
ROLES = ('progress', 'anomaly', 'waste')
TIME_FIELDS = ('best_s', 'check_s', 'expected_s', 'max_s')
SUCCESS_FIELDS = ('done', 'success')
SUBGOAL_FIELDS = ('ending', 'unit', 'found') + ROLES
SPEC_FIELDS = ('version', 'author', 'margin', 'subgoals', 'constraints')
FORMS = ('always', 'deadline', 'conditional')
SOURCES = ('human', 'plan', 'system')
STRENGTHS = ('hard', 'soft')
ON_VIOLATION = ('warn', 'check', 'hold', 'mask', 'cost', 'report_violation', 'stop', 'refuse')
CONSTRAINT_FIELDS = ('id', 'form', 'scope', 'source', 'strength', 'predicate', 'by', 'during', 'on_violation')
FACTS = ('inventory', 'health', 'day_time', 'elapsed_s', 'api_calls')
OPS = ('ge', 'le', 'eq')
MAX_MARGIN = 5.


class SpecError(ValueError):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__('; '.join(self.reasons))


@dataclass(frozen=True)
class SubgoalSpec:
    key: str
    ending: str
    unit: str
    found: str | None
    progress: tuple[str, ...]
    anomaly: tuple[str, ...]
    waste: tuple[str, ...]


@dataclass(frozen=True)
class Constraint:
    id: str
    form: str
    scope: str
    source: str
    strength: str
    predicate: dict
    by: dict | None
    during: dict | None
    on_violation: str


@dataclass(frozen=True)
class MonitorSpec:
    version: str
    author: str
    margin: float
    subgoals: dict
    constraints: tuple

    def for_step(self, skill, target):
        """The `skill:target` entry wins over the plain `skill` entry; None when the spec has neither."""
        return (self.subgoals.get(f'{skill}:{target}') if target else None) or self.subgoals.get(skill)


def constraint_priority(c):
    """System (STOP, safety) > human hard > plan-derived > soft (§8.3 rule 3); lower sorts first."""
    if c.strength == 'soft': return 3
    return {'system': 0, 'human': 1, 'plan': 2}[c.source]


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _subgoal(key, raw, out):
    where = f'subgoal {key}'
    if not isinstance(raw, dict): out.append(f'{where}: must be an object'); return None
    for field in raw:
        if field in TIME_FIELDS: out.append(f'{where}: {field} not allowed: times come from records')
        elif field in SUCCESS_FIELDS: out.append(f'{where}: {field} not allowed: caller owns success')
        elif field not in SUBGOAL_FIELDS: out.append(f'{where}: unknown field {field}')
    ending, unit, found = raw.get('ending'), raw.get('unit'), raw.get('found')
    if ending not in ENDINGS: out.append(f'{where}: ending must be one of {ENDINGS}')
    if unit not in UNITS: out.append(f'{where}: unit must be one of {UNITS}')
    if ending == 'until_found' and not (isinstance(found, str) and found): out.append(f'{where}: until_found requires found')
    if ending == 'bounded' and found is not None: out.append(f'{where}: bounded has no found')
    lists = {}
    for role in ROLES:
        names = raw.get(role, [])
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            out.append(f'{where}: {role} must be a list of signal names'); names = []
        for name in names:
            if name not in SIGNALS: out.append(f'{where}: unknown signal {name}'); continue
            actual, endings = SIGNALS[name]
            if actual != role: out.append(f'{where}: {name} is a {actual} signal, not {role}')
            elif ending in ENDINGS and ending not in endings: out.append(f'{where}: {name} not allowed for {ending}')
        lists[role] = tuple(names)
    return SubgoalSpec(key, ending, unit, found, lists['progress'], lists['anomaly'], lists['waste'])


def _predicate(where, p, out):
    """A restricted comparison over one fixed observation (§5); anything else is unsupported."""
    if not isinstance(p, dict): out.append(f'{where}: must be an object'); return
    for field in p:
        if field not in ('fact', 'op', 'value', 'item'): out.append(f'{where}: unknown field {field}')
    fact = p.get('fact')
    if fact not in FACTS: out.append(f'{where}: fact {fact!r} unsupported (no fixed observation)')
    if p.get('op') not in OPS: out.append(f'{where}: op must be one of {OPS}')
    if not _number(p.get('value')): out.append(f'{where}: value must be a finite number')
    if fact == 'inventory' and not (isinstance(p.get('item'), str) and p['item']): out.append(f'{where}: inventory requires item')
    if fact != 'inventory' and 'item' in p: out.append(f'{where}: item only applies to inventory')


def _constraint(raw, subgoals, out):
    cid = raw.get('id') if isinstance(raw, dict) else None
    where = f'constraint {cid}'
    if not isinstance(raw, dict): out.append('constraint: must be an object'); return None
    if not (isinstance(cid, str) and cid): out.append('constraint: id must be a non-empty string')
    for field in raw:
        if field not in CONSTRAINT_FIELDS: out.append(f'{where}: unknown field {field}')
    form, source, strength = raw.get('form'), raw.get('source'), raw.get('strength')
    if form not in FORMS: out.append(f'{where}: form must be one of {FORMS}')
    if source == 'llm': out.append(f'{where}: source llm rejected: constraints are written by people (decision 2026-11-05)')
    elif source not in SOURCES: out.append(f'{where}: source must be one of {SOURCES}')
    if strength not in STRENGTHS: out.append(f'{where}: strength must be one of {STRENGTHS}')
    if source in ('system', 'plan') and strength == 'soft': out.append(f'{where}: {source} constraints must be hard')
    scope = raw.get('scope')
    if scope != 'goal' and scope not in subgoals: out.append(f'{where}: scope must be goal or a subgoal key')
    if raw.get('on_violation') not in ON_VIOLATION: out.append(f'{where}: on_violation must be one of {ON_VIOLATION}')
    _predicate(f'{where} predicate', raw.get('predicate'), out)
    if form == 'deadline' and raw.get('by') is None: out.append(f'{where}: deadline requires by')
    if form == 'conditional' and raw.get('during') is None: out.append(f'{where}: conditional requires during')
    for field in ('by', 'during'):
        if raw.get(field) is not None: _predicate(f'{where} {field}', raw[field], out)
    return Constraint(cid, form, scope, source, strength, raw.get('predicate'), raw.get('by'), raw.get('during'),
                      raw.get('on_violation'))


def _interval(p):
    v = p['value']
    return {'ge': (v, math.inf), 'le': (-math.inf, v), 'eq': (v, v)}[p['op']]


def _contradictions(constraints, out):
    """Hard `always` predicates on one fact must leave some value; a hard deadline must be reachable within them."""
    def key(p): return p['fact'], p.get('item')
    bounds = {}
    for c in constraints:
        if c.strength == 'hard' and c.form == 'always':
            lo, hi = _interval(c.predicate); was = bounds.get(key(c.predicate), (-math.inf, math.inf, []))
            bounds[key(c.predicate)] = (max(was[0], lo), min(was[1], hi), was[2] + [c.id])
    for k, (lo, hi, ids) in bounds.items():
        if lo > hi: out.append(f'constraints {ids} contradict on {k}')
    for c in constraints:
        if c.strength == 'hard' and c.form == 'deadline' and key(c.predicate) in bounds:
            lo, hi = _interval(c.predicate); blo, bhi, ids = bounds[key(c.predicate)]
            if blo <= bhi and max(lo, blo) > min(hi, bhi): out.append(f'constraint {c.id} contradicts {ids}')


def parse_spec(raw):
    out = []
    if not isinstance(raw, dict): raise SpecError(['spec must be an object'])
    for field in raw:
        if field not in SPEC_FIELDS: out.append(f'unknown field {field}')
    version, author, margin = raw.get('version'), raw.get('author'), raw.get('margin')
    if not (isinstance(version, str) and version): out.append('version must be a non-empty string')
    if not (isinstance(author, str) and author): out.append('author must be a non-empty string')
    if not (_number(margin) and 0 <= margin <= MAX_MARGIN): out.append(f'margin must be a number in [0, {MAX_MARGIN}]')
    raw_subgoals = raw.get('subgoals')
    if not isinstance(raw_subgoals, dict): out.append('subgoals must be an object'); raw_subgoals = {}
    subgoals = {k: _subgoal(k, v, out) for k, v in raw_subgoals.items()}
    raw_constraints = raw.get('constraints', [])
    if not isinstance(raw_constraints, list): out.append('constraints must be a list'); raw_constraints = []
    constraints = [_constraint(c, subgoals, out) for c in raw_constraints]
    ids = [c.id for c in constraints if c]
    if len(ids) != len(set(ids)): out.append('constraint ids must be unique')
    if out: raise SpecError(out)
    _contradictions(constraints, out)
    if out: raise SpecError(out)
    return MonitorSpec(version, author, float(margin), subgoals, tuple(constraints))


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates: raise SpecError([f'duplicate key {k}' for k in duplicates])
    return dict(pairs)


def load_spec(path):
    return parse_spec(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_no_duplicates))
