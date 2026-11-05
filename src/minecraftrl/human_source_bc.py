"""Train-only recording-alias weighting, preserving within-source supervision."""
import json
import math
import re
from collections import Counter


def load_train_source_groups(root, contract, rows):
    """Use hash-bound recording aliases, not an assertion of distinct people."""
    from minecraftrl.human_navigation import checked_file
    from minecraftrl.world_store import digest
    path = checked_file(root, contract['pair_manifest'])
    pairs = json.loads(path.read_text(encoding='utf-8-sig'))['pairs']
    indexed = {}
    for pair in pairs:
        if pair['session'] in indexed:
            raise ValueError('duplicate source-group recording')
        indexed[pair['session']] = pair
    groups = []
    used = set()
    for row in rows:
        if row.get('owner') != 'human' or row.get('bc_source') != 'hindsight_reference':
            raise ValueError('source grouping requires qualified human rows')
        session = row['human_session']
        pair = indexed.get(session)
        if pair is None or pair['split'] != 'train':
            raise ValueError('source grouping cannot use development or unknown recordings')
        match = re.fullmatch(r'(.+)-[a-f0-9]{12}-\d{8}-\d{6}', session)
        if not match or not match[1].strip():
            raise ValueError('recording alias cannot be identified')
        if 'alias' in pair and pair['alias'] != match[1]:
            raise ValueError('recording alias metadata disagrees with session')
        identity = (session, row['human_frame'])
        if identity in used:
            raise ValueError('duplicate human source label')
        used.add(identity)
        groups.append(match[1])
    if not groups:
        raise ValueError('source grouping needs human train rows')
    return groups, dict(pair_manifest_sha256=digest(path), counts=dict(Counter(groups)),
        recordings=len({row['human_session'] for row in rows}), development_used=False,
        source_identity='Recording filename alias; distinct human identities not verified')


def reweight_sources(weights, is_human, human_groups, exponent=.5):
    """Allocate human mass proportional to source mass**exponent after onset weighting."""
    if type(exponent) not in (int, float) or not math.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError('source mass exponent must be finite within [0,1]')
    if (not weights or len(weights) != len(is_human) or
            any(type(flag) is not bool for flag in is_human) or
            not any(is_human) or all(is_human) or len(human_groups) != sum(is_human)):
        raise ValueError('mixed local/human rows and exact source groups required')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('source weighting needs positive finite weights')
    if any(not isinstance(group, str) or not group.strip() for group in human_groups):
        raise ValueError('nonempty source groups required')
    grouped = {}
    for weight, group in zip((w for w, flag in zip(weights, is_human) if flag), human_groups):
        grouped.setdefault(group, []).append(weight)
    before = {group: math.fsum(values) for group, values in grouped.items()}
    total = math.fsum(before.values())
    if not math.isfinite(total):
        raise ValueError('source weight sum overflow')
    identity = exponent == 1 or len(before) == 1
    if identity:
        factors = {group: 1. for group in before}
        result = list(weights)
    else:
        powered = {group: mass**exponent for group, mass in before.items()}
        normalizer = total / math.fsum(powered.values())
        factors = {group: normalizer * powered[group] / mass for group, mass in before.items()}
        groups = iter(human_groups)
        result = [w * factors[next(groups)] if flag else w for w, flag in zip(weights, is_human)]
    if any(not math.isfinite(w) or w <= 0 for w in result):
        raise ValueError('nonfinite source-adjusted weight')
    after = {group: math.fsum(w for w, g in zip(
        (w for w, flag in zip(result, is_human) if flag), human_groups) if g == group) for group in before}
    if not math.isclose(total, math.fsum(after.values()), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('source weighting changed total human mass')
    assert [w for w, flag in zip(result, is_human) if not flag] == [w for w, flag in zip(weights, is_human) if not flag]
    return result, dict(exponent=float(exponent), identity=identity,
        human_weight_before=total, human_weight_after=math.fsum(after.values()),
        source_weight_before=before, source_weight_after=after, source_factors=factors,
        local_weights_unchanged=True, within_source_ratios_preserved=True,
        normalization='Full train-set source mass after correction, human share and onset weights; no minibatch renormalization')
