"""Knowledge acquisition: LLM + web search proposes cited claims; only validated ones enter SQLite as `proposed`."""
import json
from urllib.parse import urlsplit

from .kb import CONDITION_KEYS, RELATIONS, Evidence, version
from .openai_client import output_json, url_citations, usage
from .planner import ESCAPE_METHODS, TIERS, TOOL_CLASSES, tool_of

KNOWLEDGE_PROMPT = '''You fill a Minecraft knowledge store for a planning agent. Use web search and cite every source.
Game version is given in the input; answer for that version only, and use conditions_json {"game_version": ...}
when a fact differs between versions. Return claims for the whole chain needed to obtain the goal item:
- harvest_requires for every block the agent must break, including blocks that can be broken by hand
  (object {"tool_class": "pickaxe"|"axe"|"shovel"|null, "min_tier": "hand"|"wooden"|"stone"|"iron"|"diamond"}).
  min_tier is the lowest tier that still gets the drop. If a tool is required for the drop, give its class and that
  lowest tier (e.g. a block that needs any pickaxe: {"tool_class": "pickaxe", "min_tier": "wooden"}). Use "hand" only
  when the bare hand gets the drop, and then tool_class null.
- drops for those blocks (object: the dropped item id as a JSON string).
- recipe for every crafted item on the way (object {"ingredients": {"item_id": n}, "count": n, "grid": 2|3}).
Use plain ids without the minecraft: prefix: block family "log" for any wood log, "stone", "iron_ore",
items "planks" (any wood), "stick", "crafting_table", "cobblestone", "wooden_pickaxe", "stone_pickaxe".
If the question is about movement (relation blocks_movement), answer whether solid blocks block player movement
(object true/false, subject "solid_block") and how a player removes such a block.
If the question is about leaving a pit (relation escape_by, subject "vertical_pit"), answer how a player gets out of a
1-block-wide vertical shaft deeper than one block: object {"method": "pillar_up"|"staircase"|"ladder"|"water",
"blocks_per_level": n} where pillar_up means jumping and placing a held block below, n blocks per level of height.
Every claim needs at least one source URL that you actually cited. Do not invent facts you did not find.'''

KNOWLEDGE_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['claims'],
    'properties': {'claims': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['subject', 'relation', 'object_json', 'conditions_json', 'source_urls', 'excerpt'],
        'properties': {'subject': {'type': 'string'}, 'relation': {'type': 'string', 'enum': sorted(RELATIONS)},
                       'object_json': {'type': 'string'}, 'conditions_json': {'type': 'string'},
                       'source_urls': {'type': 'array', 'items': {'type': 'string'}}, 'excerpt': {'type': 'string'}}}}}}


HYPOTHESIS_PROMPT = '''A Minecraft knowledge claim was contradicted by an observation in the running game. Use web search
and cite every source. Find an observable condition under which the claim is true but which the observed game does not
meet, using only the keys game_version, tool, enchantment, difficulty (values: a string, or for game_version
{"gte": "x.y"} / {"lte": "x.y"}). Return condition_json "null" if you find none. Do not invent facts you did not find.'''

HYPOTHESIS_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['condition_json', 'source_urls', 'excerpt'],
    'properties': {'condition_json': {'type': 'string'}, 'source_urls': {'type': 'array', 'items': {'type': 'string'}},
                   'excerpt': {'type': 'string'}}}


def acquire(client, query, context):
    """One web-search call. Returns (raw candidates, meta with usage and cited URLs)."""
    response = client.create({
        'input': [{'role': 'system', 'content': KNOWLEDGE_PROMPT},
                  {'role': 'user', 'content': json.dumps(dict(query, game_version=context.get('game_version')), ensure_ascii=False)}],
        'tools': [{'type': 'web_search'}], 'include': ['web_search_call.action.sources'],
        'text': {'format': {'type': 'json_schema', 'name': 'minecraft_knowledge', 'strict': True, 'schema': KNOWLEDGE_SCHEMA}}})
    citations = url_citations(response)
    return output_json(response)['claims'], dict(usage=usage(response), citations=sorted(citations))


def hypothesize(client, claim, observation, context):
    """One web-search call: under which observable condition does the contradicted claim hold? (spec 5.5 split)"""
    response = client.create({
        'input': [{'role': 'system', 'content': HYPOTHESIS_PROMPT},
                  {'role': 'user', 'content': json.dumps(dict(claim=claim, observation=observation,
                                                              game_version=context.get('game_version')), ensure_ascii=False)}],
        'tools': [{'type': 'web_search'}], 'include': ['web_search_call.action.sources'],
        'text': {'format': {'type': 'json_schema', 'name': 'condition_hypothesis', 'strict': True, 'schema': HYPOTHESIS_SCHEMA}}})
    return output_json(response), dict(usage=usage(response), citations=sorted(url_citations(response)))


def _condition_value_ok(key, value):
    try:
        if isinstance(value, dict):
            return key == 'game_version' and value and set(value) <= {'gte', 'lte'} and all(version(v) for v in value.values())
        if key == 'tool': return isinstance(value, str) and tool_of(value)[0] is not None  # a real tool id, no wildcard
        return isinstance(value, str) and bool(value) and (key != 'game_version' or bool(version(value)))
    except (TypeError, ValueError): return False


def validate_hypothesis(raw, citations):
    """A usable hypothesis names an observable condition and cites a consulted source."""
    cited = {_norm(u) for u in citations}
    try: condition = json.loads(raw['condition_json'])
    except (ValueError, TypeError, KeyError): return dict(condition=None, reason='bad_json')
    if not condition: return dict(condition=None, reason='no_condition')
    if not isinstance(condition, dict) or any(k not in CONDITION_KEYS for k in condition):
        return dict(condition=None, reason='unknown_condition_key')
    if not all(_condition_value_ok(k, v) for k, v in condition.items()): return dict(condition=None, reason='bad_condition')
    if not raw.get('source_urls'): return dict(condition=None, reason='no_source')
    if not all(_norm(u) in cited for u in raw['source_urls']): return dict(condition=None, reason='uncited_source')
    return dict(condition=condition, reason=None, source_urls=list(raw['source_urls']), excerpt=raw.get('excerpt', ''))


def _norm(url):
    parts = urlsplit(url.strip())
    return (parts.netloc.lower(), parts.path.rstrip('/'))


def _object_ok(relation, obj):
    if relation == 'harvest_requires':
        # A tool class with tier "hand" is ambiguous ("any pickaxe" or "no tool"): evaluation round 2 (E03) read it as
        # "no tool" and let bare hands break stone without a drop.
        return (isinstance(obj, dict) and set(obj) == {'tool_class', 'min_tier'} and obj['min_tier'] in TIERS
                and (obj['tool_class'] is None or (obj['tool_class'] in TOOL_CLASSES and obj['min_tier'] != 'hand')))
    if relation == 'drops': return isinstance(obj, str) and obj
    if relation == 'recipe':
        return (isinstance(obj, dict) and isinstance(obj.get('ingredients'), dict) and obj['ingredients']
                and all(isinstance(v, int) and v > 0 for v in obj['ingredients'].values())
                and isinstance(obj.get('count'), int) and obj['count'] > 0 and obj.get('grid') in (2, 3))
    if relation == 'blocks_movement': return isinstance(obj, bool)
    if relation == 'escape_by':
        return (isinstance(obj, dict) and set(obj) == {'method', 'blocks_per_level'} and obj['method'] in ESCAPE_METHODS
                and type(obj['blocks_per_level']) is int and 0 <= obj['blocks_per_level'] <= 4)
    return relation == 'stage_duration_s' and isinstance(obj, (int, float)) and obj > 0


def validate_candidates(raw, citations):
    """Keep only claims with a known relation, observable condition keys, a sane object and a cited source."""
    cited = {_norm(u) for u in citations}; accepted, rejected = [], []
    for c in raw:
        reason = None
        try: obj = json.loads(c['object_json']); conditions = json.loads(c['conditions_json'])
        except (ValueError, TypeError, KeyError): reason = 'bad_json'
        if reason is None and c.get('relation') not in RELATIONS: reason = 'unknown_relation'
        elif reason is None and (not isinstance(conditions, dict) or any(k not in CONDITION_KEYS for k in conditions)):
            reason = 'unknown_condition_key'
        elif reason is None and not c.get('source_urls'): reason = 'no_source'
        elif reason is None and not all(_norm(u) in cited for u in c['source_urls']): reason = 'uncited_source'
        elif reason is None and not _object_ok(c['relation'], obj): reason = 'bad_object'
        if reason: rejected.append(dict(candidate=c, reason=reason))
        else: accepted.append(dict(subject=c['subject'], relation=c['relation'], object=obj, conditions=conditions,
                                   source_urls=list(c['source_urls']), excerpt=c.get('excerpt', '')))
    return accepted, rejected


def store(kb, accepted, *, model, run_id):
    ids = []
    for c in accepted:
        evidence = [Evidence('web', 'support', url, c['excerpt'][:500]) for url in c['source_urls']]
        evidence.append(Evidence('llm_assertion', 'support', 'model:' + model))
        ids.append(kb.add_claim(c['subject'], c['relation'], c['object'], c['conditions'], evidence, run_id=run_id))
    return ids
