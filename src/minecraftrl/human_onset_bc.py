"""Train-only recorded jump-key onset weighting; no new actions or runtime rules."""
import json
import math


def jump_transitions(raw, frames):
    """Use the preceding raw frame, including frames excluded from the label mask."""
    if len(frames) != len(set(frames)):
        raise ValueError('duplicate onset classification frame')
    result = []
    for frame in frames:
        if type(frame) is not int or not 0 < frame < len(raw):
            raise ValueError('onset classification needs a preceding raw frame')
        previous, current = raw[frame - 1], raw[frame]
        if (type(previous.get('tick')) is not int or type(current.get('tick')) is not int or
                current['tick'] - previous['tick'] != 1):
            raise ValueError('onset classification requires consecutive raw ticks')
        if previous.get('isGuiOpen') is not False or current.get('isGuiOpen') is not False:
            raise ValueError('GUI transition is not a movement onset')
        keys = [row['keyboard']['keys'] for row in (previous, current)]
        if any(not isinstance(k, list) or not all(isinstance(v, str) for v in k) for k in keys):
            raise ValueError('invalid recorded keyboard state')
        held, jump = ['key.keyboard.space' in k for k in keys]
        result.append('onset' if jump and not held else 'continuation' if jump else 'nonjump')
    return result


def load_train_transitions(root, contract, rows):
    """Classify already-qualified train rows against their hash-bound original recordings."""
    from minecraftrl.human_navigation import checked_file
    from minecraftrl.world_store import digest
    path = checked_file(root, contract['pair_manifest'])
    pairs = json.loads(path.read_text(encoding='utf-8-sig'))['pairs']
    grouped = {}
    for row in rows:
        if row.get('owner') != 'human' or row.get('bc_source') != 'hindsight_reference':
            raise ValueError('onset weighting requires qualified human movement rows')
        grouped.setdefault(row['human_session'], []).append(row['human_frame'])
    classified = {}
    sources = {}
    for session, frames in grouped.items():
        matches = [pair for pair in pairs if pair['session'] == session]
        if len(matches) != 1 or matches[0]['split'] != 'train':
            raise ValueError('onset weighting cannot inspect development recordings')
        pair = matches[0]
        source = checked_file(root, dict(path=pair['jsonl'], sha256=pair['jsonl_sha256']))
        with source.open(encoding='utf-8') as stream:
            raw = [json.loads(line) for line in stream]
        kinds = jump_transitions(raw, frames)
        classified.update({(session, frame): kind for frame, kind in zip(frames, kinds)})
        sources[pair['jsonl']] = pair['jsonl_sha256']
    kinds = [classified[(row['human_session'], row['human_frame'])] for row in rows]
    return kinds, dict(pair_manifest_sha256=digest(path), source_jsonl_sha256=sources,
        counts={kind: kinds.count(kind) for kind in ('onset', 'continuation', 'nonjump')},
        recordings=len(grouped), previous_frame_source='consecutive original raw frame, not prior retained label',
        development_used=False, scope='Jump-key transition only, not physical takeoff, intent, safety or necessary action')


def reweight_onsets(weights, is_human, human_kinds, multiplier):
    """Keep local weights exact and total human weight fixed, including existing bucket weights."""
    if (type(multiplier) not in (int, float) or not math.isfinite(multiplier) or
            not 1 <= multiplier <= 8):
        raise ValueError('onset multiplier must be finite and within [1,8]')
    if (not weights or len(weights) != len(is_human) or
            any(type(flag) is not bool for flag in is_human)):
        raise ValueError('onset weighting groups must match rows')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('positive finite supervision weights required')
    if (not any(is_human) or all(is_human) or len(human_kinds) != sum(is_human) or
            any(kind not in ('onset', 'continuation', 'nonjump') for kind in human_kinds)):
        raise ValueError('mixed local/human rows and exact human classifications required')
    before = math.fsum(w for w, human in zip(weights, is_human) if human)
    if multiplier == 1:
        return list(weights), dict(multiplier=1, human_weight_before=before, human_weight_after=before,
            human_normalizer=1., local_weights_unchanged=True, default_identity=True)
    if 'onset' not in human_kinds:
        raise ValueError('no recorded onsets to upweight')
    kinds = iter(human_kinds)
    factors = [float(multiplier) if next(kinds) == 'onset' else 1. for human in is_human if human]
    weighted_total = math.fsum(w * factor for w, factor in zip(
        (w for w, human in zip(weights, is_human) if human), factors))
    normalizer = before / weighted_total
    factors = iter(factors)
    result = [w * next(factors) * normalizer if human else w for w, human in zip(weights, is_human)]
    after = math.fsum(w for w, human in zip(result, is_human) if human)
    if not math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('onset weighting changed the total human loss weight')
    assert [w for w, human in zip(result, is_human) if not human] == [w for w, human in zip(weights, is_human) if not human]
    return result, dict(multiplier=float(multiplier), human_weight_before=before, human_weight_after=after,
        human_normalizer=normalizer, local_weights_unchanged=True, default_identity=False,
        original_bucket_weighting_retained=True, normalization='weighted total over train human rows; not minibatch normalization')
