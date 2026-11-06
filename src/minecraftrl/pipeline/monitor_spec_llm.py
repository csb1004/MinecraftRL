"""LLM proposal of a goal monitor spec (plan docs/specs/2026-11-05-goal-monitor-spec-plan.md §1, §4; task D 2026-11-07).

The LLM sees the goal, the planned sub-goals as caller-owned facts (skill kind, target, target block, whether the
target's location is known) and the fixed signal list. Per sub-goal it returns the ending, unit, found block and
signal names. The program builds the spec (version, author, margin and constraints are the caller's), then applies
the fact rules here and the same parse_spec checks a human spec gets. One repair round sends the reasons back.
The LLM never gives times, success conditions or constraints (people write constraints, decision 2026-11-05).
"""
import json
from dataclasses import asdict, dataclass

from .monitor_spec import ENDINGS, SIGNALS, UNITS, SpecError, parse_spec
from .openai_client import output_json, usage

SIGNAL_TEXT = {
    'objective_inventory_increased': 'the target item count in the server inventory rose',
    'new_position_cell': 'the player entered a 2-block position cell not visited before',
    'new_search_view': 'the player looked in one of 8 directions not yet seen from this cell',
    'rise': 'the player is higher than before (server y)',
    'no_progress': 'no new cell, view or target evidence for a while',
    'repeated_cycle': 'the player keeps walking the same loop of positions',
    'stationary_repetition': 'the player stays in place repeating the same keys',
    'harvest_without_drop': 'a block broke but nothing entered the inventory',
    'needed_tool_lost': 'a tool or item the plan needs left the inventory',
    'low_health': 'server health is low',
    'blocked_overhead': 'a block above the head stops climbing',
    'climb_without_block': 'trying to pillar up with no placeable block in hand',
    'moving_but_stuck': 'movement keys are pressed but the position barely changes for about 2 seconds',
    'looking_down': 'the view pitch is steeply down',
    'revisiting_area': 'walking back into a cell already visited and left',
}
UNIT_TEXT = {'step': 'the whole step', 'item': 'each item of the target count', 'block': 'each block of height climbed'}

PROMPT = '''You write a monitor spec for a Minecraft agent. A goal is split into planned sub-goals. For each sub-goal choose:
- ending: "bounded" when the target's location is known and the work takes a steady time; "until_found" when the
  target must be searched for (no time limit; the monitor watches whether exploration continues). A sub-goal whose
  target_location is "unknown" must be until_found; "known" must be bounded.
- unit: what the duration scales with: {units}.
- found: for until_found only, the block id whose sighting ends the search (the sub-goal's target block); null for bounded.
- progress, anomaly, waste: names from the fixed signal list below, each used only in its own role and only if its
  allowed endings include the chosen ending. Pick the signals that fit what the sub-goal does; lists may be empty.
- reason: one short sentence on why.
Do not give times, success conditions or constraints: times come from recorded runs, the caller owns success, and
people write constraints. Return every sub-goal key exactly once.

Fixed signals (name | role | allowed endings | meaning):
{signals}'''

_ITEM_PROPS = {
    'key': {'type': 'string'}, 'ending': {'type': 'string', 'enum': list(ENDINGS)},
    'unit': {'type': 'string', 'enum': list(UNITS)}, 'found': {'type': ['string', 'null']},
    **{role: {'type': 'array', 'items': {'type': 'string', 'enum': sorted(SIGNALS)}} for role in ('progress', 'anomaly', 'waste')},
    'reason': {'type': 'string'}}
PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['subgoals'],
    'properties': {'subgoals': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                                           'required': list(_ITEM_PROPS), 'properties': _ITEM_PROPS}}}}


@dataclass(frozen=True)
class SubgoalFact:
    """What the program knows about a planned sub-goal (caller-owned, not from the LLM)."""
    key: str
    kind: str                 # 'policy' or 'rule'
    target: str
    block: str | None         # the block a policy step works on; None for rule steps
    location: str             # 'known' or 'unknown'
    description: str


def system_prompt():
    signals = '\n'.join(f'- {name} | {role} | {"/".join(endings)} | {SIGNAL_TEXT[name]}'
                        for name, (role, endings) in SIGNALS.items())
    units = '; '.join(f'"{u}" = {t}' for u, t in UNIT_TEXT.items())
    return PROMPT.format(units=units, signals=signals)


def proposal_payload(goal, facts, previous=None, reasons=None):
    """Responses API payload; with `previous` and `reasons` it is the repair round."""
    user = json.dumps({'goal': goal, 'subgoals': [asdict(f) for f in facts]}, ensure_ascii=False)
    messages = [{'role': 'system', 'content': system_prompt()}, {'role': 'user', 'content': user}]
    if previous is not None:
        messages += [{'role': 'assistant', 'content': json.dumps(previous, ensure_ascii=False)},
                     {'role': 'user', 'content': 'The program rejected the proposal:\n- ' + '\n- '.join(reasons) +
                      '\nReturn a corrected full proposal.'}]
    return {'input': messages,
            'text': {'format': {'type': 'json_schema', 'name': 'monitor_spec', 'strict': True, 'schema': PROPOSAL_SCHEMA}}}


def spec_from_proposal(proposal, *, version, author, margin, constraints):
    """The spec as the program writes it: only the per-sub-goal choices come from the proposal."""
    subgoals = {}
    for p in proposal.get('subgoals', []):
        entry = {'ending': p.get('ending'), 'unit': p.get('unit'),
                 **{role: list(p.get(role) or []) for role in ('progress', 'anomaly', 'waste')}}
        if p.get('found') is not None: entry['found'] = p['found']
        subgoals[p.get('key')] = entry
    return {'version': version, 'author': author, 'margin': margin, 'subgoals': subgoals, 'constraints': list(constraints)}


def check_facts(proposal, facts):
    """Program rules the plan fixes: every planned key exactly once; unknown location -> until_found with the
    target block as found; known location -> bounded."""
    out, by_key = [], {f.key: f for f in facts}
    keys = [p.get('key') for p in proposal.get('subgoals', [])]
    out += [f'duplicate subgoal {k}' for k in sorted({k for k in keys if keys.count(k) > 1}, key=str)]
    out += [f'missing subgoal {k}' for k in by_key if k not in keys]
    out += [f'unknown subgoal {k}' for k in dict.fromkeys(keys) if k not in by_key]
    for p in proposal.get('subgoals', []):
        fact = by_key.get(p.get('key'))
        if fact is None: continue
        if fact.location == 'unknown' and p.get('ending') != 'until_found':
            out.append(f'subgoal {fact.key}: target location unknown, ending must be until_found')
        if fact.location == 'known' and p.get('ending') != 'bounded':
            out.append(f'subgoal {fact.key}: target location known, ending must be bounded')
        if p.get('ending') == 'until_found' and p.get('found') != fact.block:
            out.append(f'subgoal {fact.key}: found must be {fact.block}')
    return out


def _check(proposal, facts, **spec_fields):
    reasons = check_facts(proposal, facts)
    raw = spec_from_proposal(proposal, **spec_fields)
    try: parse_spec(raw)
    except SpecError as e: reasons += e.reasons
    return raw, reasons


def propose_spec(client, goal, facts, *, version, margin, constraints, repairs=1):
    """-> dict(ok, spec (raw, or None), reasons, attempts=[{proposal, reasons, usage}]). One API call per attempt."""
    spec_fields = dict(version=version, author='llm:' + client.model, margin=margin, constraints=constraints)
    attempts, previous, reasons = [], None, []
    for _ in range(repairs + 1):
        response = client.create(proposal_payload(goal, facts, previous, reasons if previous is not None else None))
        proposal = output_json(response)
        raw, reasons = _check(proposal, facts, **spec_fields)
        attempts.append(dict(proposal=proposal, reasons=reasons, usage=usage(response)))
        if not reasons: return dict(ok=True, spec=raw, reasons=[], attempts=attempts)
        previous = proposal
    return dict(ok=False, spec=None, reasons=reasons, attempts=attempts)
