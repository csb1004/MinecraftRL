"""Remedies for a stalled or repeatedly failing step (MineEvolve Inducer + Curator; design 2026-11-25-stagnation-remedy).

The knowledge store is asked first (relation 'remedy', subject '<failure type>:<skill>'). Without a usable, executable
remedy an approved LLM proposes one repair from a fixed vocabulary of actions the session can execute; the Curator
checks schema, match, executability, specificity and conflict (MineEvolve eq. 6) before it is stored as 'proposed'.
The game decides afterwards: the repaired attempt's result is support or refutation. Without an LLM, or when its
proposal is rejected, a temporary turn-away is used for this hold only and nothing is stored.
"""
import json

from .feedback import climbing
from .kb import USABLE, Evidence
from .openai_client import output_json, usage
from .planner import placeable, tool_of

ACTIONS = {
    'turn_away': 'turn 180 degrees and level the view, then try the step again (no argument)',
    'level_view': 'level the view, then try the step again (no argument)',
    'switch_skill': 'try the step again with another registered skill for the same target (argument: one of alternatives)',
    'equip': 'hold another item from the inventory, then try again (argument: an inventory item not already held; a '
             'full block such as dirt, cobblestone or planks while climbing, a tool otherwise)',
    'clear_above': 'clear the blocks over the head, then climb the remaining height (climbing steps only, no argument)',
    'replan': 'plan the rest of the goal again from the current inventory (no argument)',
}
FIELDS = ('failure_type', 'skill', 'target', 'climbing', 'held')  # the state fields a trigger may name
NO_ARGUMENT = ('turn_away', 'level_view', 'clear_above', 'replan')
TEMPORARY = dict(action='turn_away', arg=None, risk_pattern='no stored or induced remedy for this trigger; turn and try again')

INDUCER_PROMPT = '''A step of a Minecraft agent stalled or keeps failing. Its low-level policy (STEVE-1) is fixed; you choose
one repair the program can execute before the step is tried again. Use only the listed actions, with an argument only
where the action takes one and the situation allows it: equip only an item in the inventory that is not already held,
switch_skill only to a listed alternative, clear_above only while climbing. Plain retry is not available. Name the
trigger with situation fields that say when this repair applies (failure_type is required; use null for fields that do
not matter), describe in one sentence the risk pattern you see in the feedback, and give a short reason. Do not repeat
a repair listed as refuted.'''

INDUCER_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['action', 'arg', 'risk_pattern', 'trigger', 'reason'],
    'properties': {
        'action': {'type': 'string', 'enum': sorted(ACTIONS)}, 'arg': {'type': ['string', 'null']},
        'risk_pattern': {'type': 'string'}, 'reason': {'type': 'string'},
        'trigger': {'type': 'object', 'additionalProperties': False, 'required': list(FIELDS), 'properties': {
            'failure_type': {'type': 'string'}, 'skill': {'type': ['string', 'null']}, 'target': {'type': ['string', 'null']},
            'climbing': {'type': ['boolean', 'null']}, 'held': {'type': ['string', 'null']}}}}}


def trigger_key(failure_type, skill): return '%s:%s' % (failure_type, skill)


def situation(step, failure_type, inventory, held, skills):
    """The fixed state fields a remedy is matched and checked against."""
    alternatives = sorted(name for name, spec in skills.items()
                          if name != step.skill and step.target in (spec.get('target_items') or ()))
    return dict(failure_type=failure_type, skill=step.skill, target=step.target, climbing=climbing(step), held=held or 'air',
                inventory=dict(inventory), alternatives=alternatives)


def executable(action, arg, sit):
    if action in NO_ARGUMENT:
        return arg is None and (action != 'clear_above' or sit['climbing'])
    if action == 'switch_skill': return arg in sit['alternatives']
    if action == 'equip' and isinstance(arg, str):
        # A climb pillars with a full block; a harvest step holds a tool (the guard blocks a bare-hand or wrong-tool hit).
        fits = placeable(arg) if sit['climbing'] else tool_of(arg)[0] is not None
        return sit['inventory'].get(arg, 0) > 0 and fits
    return False


def curate(candidate, sit, refuted):
    """MineEvolve eq. 6 -> (accepted, failed check or None)."""
    if (not isinstance(candidate, dict) or candidate.get('action') not in ACTIONS or not isinstance(candidate.get('trigger'), dict)
            or not isinstance(candidate.get('risk_pattern'), str)): return False, 'schema'
    trigger = candidate['trigger']
    if ('failure_type' not in trigger or any(k not in FIELDS for k in trigger)
            or any(sit.get(k) != v for k, v in trigger.items())): return False, 'match'
    action, arg = candidate['action'], candidate.get('arg')
    if not executable(action, arg, sit): return False, 'exec'
    if (action == 'equip' and arg == sit['held']) or len(candidate['risk_pattern'].split()) < 4: return False, 'spec'
    if (action, arg) in refuted: return False, 'conflict'
    return True, None


def store(kb, key, candidate, context, *, model, run_id):
    obj = dict(action=candidate['action'], arg=candidate.get('arg'), risk_pattern=candidate['risk_pattern'],
               trigger=dict(candidate['trigger']))
    ev = Evidence('llm_assertion', 'support', 'model:' + model, str(candidate.get('reason', ''))[:500])
    return kb.add_claim(key, 'remedy', obj, {'game_version': context['game_version']}, [ev], run_id=run_id)


def confidence(claim): return (claim.support + 1) / (claim.support + claim.refute + 2)


def retrieve(kb, key, context, sit):
    """The most trusted usable remedy for this trigger that matches and can run now, or None."""
    for claim in sorted(kb.query(key, 'remedy', context=context), key=lambda c: (-confidence(c), -c.id)):
        obj = claim.object
        if all(sit.get(k) == v for k, v in obj.get('trigger', {}).items()) and executable(obj['action'], obj.get('arg'), sit):
            return claim
    return None


def refuted(kb, key, context):
    return {(c.object['action'], c.object.get('arg')) for c in kb.query(key, 'remedy', context=context, usable_only=False)
            if c.status not in USABLE}


def outcome(kb, claim_id, success, ref, *, run_id):
    """The repaired attempt's result is game evidence; a remedy that did not repair is disputed (no longer retrieved)."""
    kb.add_evidence(claim_id, Evidence('game_observation', 'support' if success else 'refute', ref,
                                       'repaired attempt completed' if success else 'repaired attempt failed again'), run_id=run_id)
    if not success: kb.set_status(claim_id, 'disputed', 'remedy did not repair the step', run_id=run_id)


def induce(client, sit, feedback, avoid):
    response = client.create({
        'input': [{'role': 'system', 'content': INDUCER_PROMPT},
                  {'role': 'user', 'content': json.dumps(dict(situation=sit, feedback=feedback, actions=ACTIONS,
                                                              refuted=[list(r) for r in sorted(avoid, key=str)]), ensure_ascii=False)}],
        'text': {'format': {'type': 'json_schema', 'name': 'remedy', 'strict': True, 'schema': INDUCER_SCHEMA}}})
    return output_json(response), usage(response)


def choose(kb, llm, context, sit, feedback, *, run_id):
    """-> dict(source 'knowledge'|'induced'|'temporary', remedy, claim_id, rejected, usage)."""
    key = trigger_key(sit['failure_type'], sit['skill'])
    claim = retrieve(kb, key, context, sit)
    if claim is not None:
        return dict(source='knowledge', remedy=dict(claim.object), claim_id=claim.id, rejected=None, usage=None)
    rejected = None; spent = None
    if llm is not None:
        avoid = refuted(kb, key, context)
        try: candidate, spent = induce(llm, sit, feedback, avoid)
        except (RuntimeError, ValueError) as exc: rejected = 'llm_failed: ' + str(exc)
        else:
            if isinstance(candidate, dict) and isinstance(candidate.get('trigger'), dict):
                candidate['trigger'] = {k: v for k, v in candidate['trigger'].items() if v is not None}
            ok, rejected = curate(candidate, sit, avoid)
            if ok:
                cid = store(kb, key, candidate, context, model=llm.model, run_id=run_id)
                chosen = dict(action=candidate['action'], arg=candidate.get('arg'), risk_pattern=candidate['risk_pattern'],
                              trigger=candidate['trigger'])
                return dict(source='induced', remedy=chosen, claim_id=cid, rejected=None, usage=spent)
    return dict(source='temporary', remedy=dict(TEMPORARY, trigger=dict(failure_type=sit['failure_type'], skill=sit['skill'])),
                claim_id=None, rejected=rejected, usage=spent)
