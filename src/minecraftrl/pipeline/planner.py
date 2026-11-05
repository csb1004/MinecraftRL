"""Rule-based planner: work back from the goal through usable knowledge only (never asks the LLM).

No knowledge is not "no condition": without a usable harvest requirement the plan is naive and the guard checks
at the last reversible moment. The plan also carries what the goal still needs, so monitors follow the goal (RM).
"""
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field

from .kb import USABLE
from .steps import LOGS, Step

TIERS = ('hand', 'wooden', 'stone', 'iron', 'diamond')
TOOL_CLASSES = ('pickaxe', 'axe', 'shovel', 'hoe', 'sword')
HARVEST_SKILLS = {'log': 'collect_log', 'stone': 'mine_cobblestone', 'iron_ore': 'mine_iron_ore'}
# Abstract policy skill -> group of configs/iron-demo.json `selected` (the gate-chosen prompt).
# 'escape_pit' (pit-escape probe 2026-11-04: `climb out of the hole` 3/3) climbs a 1-wide pit with the held block stack.
SKILL_GROUPS = {'collect_log': 'log', 'mine_cobblestone': 'cobblestone', 'mine_iron_ore': 'iron_ore', 'escape_pit': 'escape'}
# Disclosed rule skills: craft (and place/reuse a table), equip an owned item into the hand, clear the blocks over the
# head (STEVE-1 prompts cleared 0/9 ceilings in results/pit-ceiling-probe-20261104; a learning request is recorded).
RULE_SKILLS = ('craft', 'equip', 'clear_above')
# escape_by methods the knowledge store accepts; only pillar_up has an executable skill.
ESCAPE_METHODS = ('pillar_up', 'staircase', 'ladder', 'water')
PILLAR_BLOCKS = ('dirt', 'cobblestone', 'stone', 'netherrack')
# Declared effects of registered harvest skills (configs/iron-demo.json target_items): what a skill is for, not world
# knowledge about drops. Used only to start a naive plan for such a goal item; the condition stays unverified.
SKILL_EFFECTS = {'cobblestone': 'stone'}


@dataclass
class Plan:
    id: str
    steps: list
    naive: bool
    problems: list
    guarded: frozenset
    needs: frozenset
    missing: tuple = ()  # subjects whose knowledge the plan lacks (asked again by the slow path)
    keep: dict = field(default_factory=dict)  # owned items the plan's crafts will consume (never spent elsewhere)


def block_family(block_id): return 'log' if block_id.endswith('_log') else block_id


def placeable(item):
    """Full blocks a pillar may use (the program's list; workstations and tools are never pillar material)."""
    return item in PILLAR_BLOCKS or item.endswith('_planks') or item.endswith('_log')


def tool_of(item_id):
    for cls in TOOL_CLASSES:
        tier = item_id[:-len(cls) - 1] if item_id.endswith('_' + cls) else None
        if tier in TIERS: return cls, tier
    return None, 'hand'


def harvest_verdict(claims, held_item):
    """'unknown' without usable knowledge; else the strictest requirement decides. Server harvestability is not an input."""
    usable = [c for c in claims if c.status in USABLE]
    if not usable: return 'unknown', None
    need = max((c.object for c in usable), key=lambda r: TIERS.index(r['min_tier']))
    cls, tier = tool_of(held_item)
    fits = TIERS.index(tier) >= TIERS.index(need['min_tier']) and (need['tool_class'] is None or need['tool_class'] == cls)
    return ('ok' if fits or need['min_tier'] == 'hand' else 'violation'), need


def oversight(goal_item, steps, tools=()):
    """(guarded block families, needed items) for the current goal: what the monitors protect."""
    guarded = {block_family(goal_item)} | {block_family(s.block) for s in steps if s.kind == 'policy' and s.block}
    # Escape steps spend blocks and gain height: neither is a goal resource.
    needs = {goal_item} | {s.target for s in steps if s.skill != 'equip' and s.target != 'rise'} | set(tools)
    return frozenset(guarded), frozenset(needs)


def concrete_skill(skill, selected):
    if skill in RULE_SKILLS or skill not in SKILL_GROUPS: return skill  # rule skill, or already a concrete prompt
    return selected[SKILL_GROUPS[skill]]


def clear_plan(kb, context):
    """(steps, problems, missing) to remove what blocks the head while climbing: only with usable knowledge that solid
    blocks block movement (user flow 2026-11-04: head hits -> knowledge -> clear -> climb)."""
    claims = [c for c in kb.query('solid_block', 'blocks_movement', context=context) if c.object is True]
    if not claims: return [], ['no usable blocks_movement knowledge for solid_block'], ('solid_block',)
    return [Step('clear-above', 'clear_above', 'rule', 'cleared', 1, claim_ids=(claims[0].id,),
                 params={'goals': ['cleared'], 'limit_blocks': 4})], [], ()


def validate_plan(steps, kb, context):
    known = set(HARVEST_SKILLS.values()) | set(SKILL_GROUPS) | set(RULE_SKILLS)
    return ['unknown skill: %s' % s.skill for s in steps if s.skill not in known]


def plan(goal_item, goal_count, inventory, kb, context, *, workstation_margin_logs=1):
    inv = Counter(inventory); inv['log'] = sum(inv.get(k, 0) for k in LOGS)
    q = lambda subject, relation: kb.query(subject, relation, context=context)
    problems, order, owned, recipe_ids, missing = [], [], [], {}, []
    total = Counter()  # every demand addition, to know which owned items the crafts consume

    def lack(item, text):
        problems.append(text)
        if item not in missing: missing.append(item)

    def source(item):
        for family in HARVEST_SKILLS:
            for claim in q(family, 'drops'):
                if claim.object == item: return family, (claim.id,)
        if item in HARVEST_SKILLS: return item, ()
        # A registered skill's declared effect (round 3c: the LLM kept answering "cobblestone drops cobblestone"); the
        # block's harvest condition still comes from knowledge or the guard asks at the last moment.
        return (SKILL_EFFECTS[item], ()) if item in SKILL_EFFECTS else (None, ())

    def requirement(family):
        claims = q(family, 'harvest_requires')
        if not claims: return None, ()
        strict = max(claims, key=lambda c: TIERS.index(c.object['min_tier']))
        return strict.object, (strict.id,)

    def recipe(item):
        claims = q(item, 'recipe')
        return claims[0] if claims and source(item)[0] is None else None

    def bases(item):
        r = recipe(item)
        return [b for ing in r.object['ingredients'] for b in bases(ing)] if r else [item]

    def tool_for(family):
        need, _ = requirement(family)
        if need and need['tool_class'] and need['min_tier'] != 'hand': ensure('%s_%s' % (need['min_tier'], need['tool_class']))

    def ensure(tool):
        if tool in order or tool in owned: return
        if inv.get(tool, 0) >= 1: owned.append(tool); return
        r = recipe(tool)
        if r is None: lack(tool, 'no usable recipe for required tool %s' % tool); return
        recipe_ids[tool] = [r.id]
        for ing in r.object['ingredients']:
            for base in bases(ing):
                family = source(base)[0]
                if family is not None and family != 'log': tool_for(family)
        order.append(tool)

    family, drop_ids = source(goal_item); naive_params = {'precondition': 'unverified'}
    if goal_item in SKILL_EFFECTS and not drop_ids and family == SKILL_EFFECTS[goal_item]:
        naive_params = dict(naive_params, source='skill_effect')
        if not q(family, 'drops'): lack(family, 'no usable drops knowledge for %s' % family)
    goal_step = lambda claim_ids, params: Step('mine-' + goal_item, HARVEST_SKILLS[family], 'policy', goal_item, goal_count,
                                               block=family, claim_ids=claim_ids, params=params)
    if family is None:
        problems.append('no harvest skill for %s' % goal_item)
        return finish(goal_item, [], True, problems, owned)
    need, need_ids = requirement(family)
    if need is None:
        lack(family, 'no usable harvest_requires for %s' % family)
        return finish(goal_item, [goal_step((), naive_params)], True, problems, owned, missing)
    tool_for(family)

    # Aggregate intermediate demand across all tool crafts before rounding (sticks for both pickaxes share one craft).
    demand, mined, grid3 = Counter(), {}, False
    for tool in order:
        r = recipe(tool).object; grid3 |= r['grid'] == 3
        demand.update(r['ingredients']); total.update(r['ingredients'])
        mined[tool] = Counter({i: n for i, n in r['ingredients'].items() if recipe(i) is None and source(i)[0] not in (None, 'log')})
    if grid3 and inv.get('crafting_table', 0) < 1: demand['crafting_table'] += 1
    if grid3: total['crafting_table'] += 1  # placed for the 3x3 craft
    while True:
        pending = [i for i in demand if recipe(i) and i not in order]
        if not pending: break
        # Expand an intermediate that no other pending intermediate still needs.
        item = next(i for i in pending if not any(i in recipe(j).object['ingredients'] for j in pending if j != i))
        r = recipe(item); count = demand.pop(item) - inv.get(item, 0)
        for tool in order:
            if item in recipe(tool).object['ingredients'] or item == 'crafting_table': recipe_ids[tool].append(r.id)
        if count > 0:
            for ing, k in r.object['ingredients'].items():
                n = math.ceil(count / r.object['count']) * k; demand[ing] += n; total[ing] += n
    for item in demand:  # what is left must be mined; an unknown intermediate is a gap, not free
        if item not in order and source(item)[0] is None: lack(item, 'no usable recipe or source for %s' % item)
    steps = []
    between = any(sum(mined[t].values()) for t in order[1:])
    logs = demand.get('log', 0) - inv['log'] + (workstation_margin_logs if grid3 and between else 0)
    if logs > 0:
        steps.append(Step('logs', HARVEST_SKILLS['log'], 'policy', 'log', logs, block='log', claim_ids=requirement('log')[1]))
    for tool in order:
        for item, n in mined[tool].items():
            if n - inv.get(item, 0) > 0:
                fam, ids = source(item)
                steps.append(Step('mine-' + item, HARVEST_SKILLS[fam], 'policy', item, n - inv.get(item, 0), block=fam,
                                  claim_ids=requirement(fam)[1] + ids))
        steps.append(Step('craft-' + tool, 'craft', 'rule', tool, 1, claim_ids=tuple(dict.fromkeys(recipe_ids[tool])),
                          params={'goals': [tool], 'equip': tool}))
    steps.append(goal_step(need_ids + drop_ids, {}))
    keep = {item: min(inv[item], n) for item, n in total.items() if inv.get(item, 0) > 0}
    return finish(goal_item, steps, False, problems, owned, missing, keep)


def plan_id(steps):
    return 'plan-' + hashlib.sha256(json.dumps([asdict(s) for s in steps], sort_keys=True).encode()).hexdigest()[:12]


def finish(goal_item, steps, naive, problems, owned, missing=(), keep=None):
    guarded, needs = oversight(goal_item, steps, owned)
    return Plan(plan_id(steps), steps, naive, problems, guarded, needs, tuple(missing), dict(keep or {}))


def escape_plan(goal_item, goal_count, inventory, depth, kb, context):
    """Escape sub-goal for a 1-wide pit (user decision 2026-11-04, spec 5.3), then the rest of the goal.

    Uses only usable `vertical_pit escape_by` knowledge; pillar_up needs (levels - 1) * blocks_per_level blocks. The
    pillar never spends what the rest of the plan consumes; each spare stack gets an equip step and a climb segment
    (the hand cannot switch stacks under the tool lock)."""
    rest = plan(goal_item, goal_count, inventory, kb, context)
    problems, missing, steps = list(rest.problems), list(rest.missing), []
    claims = kb.query('vertical_pit', 'escape_by', context=context)
    pillar = [c for c in claims if c.object['method'] == 'pillar_up']
    if not claims:
        problems.insert(0, 'no usable escape_by knowledge for vertical_pit'); missing.insert(0, 'vertical_pit')
    elif not pillar:
        problems.insert(0, 'no escape skill for method(s): ' + ', '.join(sorted({c.object['method'] for c in claims})))
    else:
        claim = pillar[0]; per = claim.object['blocks_per_level']; levels = max(1, int(round(depth)))
        need = (levels - 1) * per
        kept = lambda item: (item in rest.keep or item in rest.needs or (item.endswith('_log') and 'log' in rest.keep)
                             or (item.endswith('_planks') and 'planks' in rest.keep))
        spare = sorted(((int(n), item) for item, n in inventory.items() if n > 0 and placeable(item) and not kept(item)),
                       key=lambda pair: (-pair[0], pair[1]))
        chosen, have = [], 0
        for n, item in spare:
            if have >= need: break
            chosen.append((item, n)); have += n
        climb = lambda i, rise: Step('climb-%d' % i, 'escape_pit', 'policy', 'rise', rise, claim_ids=(claim.id,),
                                     params={'until': 'rise', 'expect': 'rise', 'to_surface': True})
        if need == 0: steps = [climb(1, levels)]
        elif have < need:
            problems.insert(0, 'not enough spare placeable blocks to pillar %d levels: %d of %d' % (levels, have, need))
        else:
            remaining = levels
            for i, (item, n) in enumerate(chosen, 1):
                rise = remaining if i == len(chosen) else min(n // per, remaining)
                # Only the climb depends on the escape knowledge (an equip success is no evidence for it).
                steps += [Step('equip-' + item, 'equip', 'rule', item, 1, params={'goals': [item], 'equip': item, 'equip_only': True}),
                          climb(i, rise)]
                remaining -= rise
    steps += rest.steps
    return Plan(plan_id(steps), steps, rest.naive, problems, rest.guarded, rest.needs, tuple(missing), rest.keep)
