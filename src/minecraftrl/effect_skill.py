"""Backend-neutral requests and committed inventory effects, separate from root RM reward."""
from copy import deepcopy
import math

from .goal_contract import digest
from .goal_spec import assert_goal
from .logo_validators import observation_valid
from .rl_core import gae
from .runtime import inventory_counts

SCHEMA = 'effect_skill_v1'
EFFECT = 'inventory_above_credited_peak'
EXTRA_FIELDS = ('effect_gap_log2', 'credited_progress_fraction')


def effect_context(request, count):
    gap = max(0, request['threshold']-count)
    return [math.log2(1+gap)/math.log2(65.),
            min(1., (request['credited_peak']-request['root_baseline'])/request['quantity'])]


class EffectLedger:
    """One live request; only a committed authoritative transition can earn credit."""
    def __init__(self):
        self.peaks = {}
        self.request = None
        self.sequence = 0
        self.last_tick = self.last_count = None

    def begin(self, queue):
        key = queue.active
        if key is None or queue.goal_spec is None:
            raise ValueError('active canonical root goal required')
        contract = assert_goal(queue, key)
        record = queue.records[key]
        if self.request is not None:
            if (self.request['goal_id'] != key or self.request['revision'] != record['revision']):
                raise ValueError('live skill goal/revision changed')
        else:
            baseline = record['baseline']
            peak = self.peaks.setdefault(key, baseline)
            self.sequence += 1
            request = dict(schema=SCHEMA, effect=EFFECT, sequence=self.sequence,
                goal_id=key, revision=record['revision'], goal_contract_sha256=contract['sha256'],
                source_sha256=queue.goal_spec['source_sha256'], items=contract['items'],
                target_ref=contract['target_ref'], quantity=contract['quantity'],
                root_baseline=baseline, credited_peak=peak, threshold=peak+1,
                started_tick=queue.last_tick, evidence='authoritative_server_inventory')
            self.request = dict(request, request_id=digest(request))
        if self.last_tick is not None and queue.last_tick != self.last_tick:
            raise ValueError('effect ledger and committed observation differ')
        self.last_tick = queue.last_tick
        self.last_count = sum(queue.inventory.get(n, 0) for n in self.request['items'])
        return deepcopy(self.request)

    def commit(self, request, server, *, terminal):
        if request != self.request or self.request is None:
            raise ValueError('stale or unissued effect request')
        if not observation_valid(server) or server['server_tick'] != self.last_tick+1:
            raise ValueError('one authoritative committed server tick required')
        inventory = inventory_counts({'inventory': server.get('server_inventory')})
        count = sum(inventory.get(n, 0) for n in request['items'])
        success = count >= request['threshold']
        result = dict(request_id=request['request_id'], before_count=self.last_count,
            after_count=count, before_tick=self.last_tick, after_tick=int(server['server_tick']),
            source='authoritative_server_inventory', skill_reward=float(success),
            skill_terminal=bool(success or terminal), success=success)
        self.last_tick, self.last_count = int(server['server_tick']), count
        if success:
            self.peaks[request['goal_id']] = count
        if success or terminal:
            self.request = None
        return result


def skill_returns(run, *, gamma=.999):
    """GAE stops at effect success even when the root goal/game continues."""
    advantages, returns, segment = [], [], []
    rows = run['rows']
    for index, row in enumerate(rows):
        segment.append(row)
        last = index == len(rows)-1
        terminal = bool(row['skill_terminal'] or (last and run['terminal']))
        if terminal or last:
            a, r = gae([s['skill_reward'] for s in segment], [s['value'] for s in segment],
                0. if terminal else run['bootstrap'], terminal=terminal, gamma=gamma, lam=1.)
            advantages.extend(a); returns.extend(r); segment = []
    return advantages, returns


def validate_effect_trace(data):
    """Reconstruct every request/credit; raw-server evidence is audited independently."""
    compiled = data.get('goal_spec')
    if compiled is None:
        raise ValueError('effect trace needs its canonical goal specification')
    peaks, baselines, completed = {}, {}, set()
    live = None
    sequence = 0
    previous = None
    for row in data['rows']:
        req, out = row['effect_request'], row['effect_outcome']
        key = row['goal_id']
        c = compiled['contracts'][key]
        integers = [req[k] for k in ('root_baseline','credited_peak','threshold','started_tick','revision')]
        integers += [out[k] for k in ('before_count','after_count','before_tick','after_tick')]
        if any(type(v) is not int or v < 0 for v in integers) or req['revision'] < 1:
            raise ValueError('invalid effect evidence counts/revision')
        if key in completed:
            raise ValueError('effect action after root completion')
        if previous is not None and out['before_tick'] != previous['effect_outcome']['after_tick']:
            raise ValueError('effect tick discontinuity')
        if previous is not None and previous['goal_id'] == key:
            if out['before_count'] != previous['effect_outcome']['after_count']:
                raise ValueError('effect inventory discontinuity')
        elif out['before_count'] != req['root_baseline'] or (previous and not previous['reward']):
            raise ValueError('invalid root activation boundary')
        if live is None:
            sequence += 1
            baseline = baselines.setdefault(key, req['root_baseline'])
            peak = peaks.setdefault(key, baseline)
            expected = dict(schema=SCHEMA, effect=EFFECT, sequence=sequence, goal_id=key,
                revision=req['revision'], goal_contract_sha256=c['sha256'],
                source_sha256=compiled['source_sha256'], items=c['items'], target_ref=c['target_ref'],
                quantity=c['quantity'], root_baseline=baseline, credited_peak=peak, threshold=peak+1,
                started_tick=out['before_tick'], evidence='authoritative_server_inventory')
            live = dict(expected, request_id=digest(expected))
        if req != live or out['request_id'] != req['request_id']:
            raise ValueError('effect request identity/credit mismatch')
        success = out['after_count'] >= req['threshold']
        terminal = bool(success or row['terminal'])
        if (out['source'] != 'authoritative_server_inventory' or out['after_tick'] != out['before_tick']+1
                or out['success'] != success or out['skill_reward'] != float(success)
                or row['skill_reward'] != float(success) or row['skill_terminal'] != terminal
                or out['skill_terminal'] != terminal):
            raise ValueError('effect reward/boundary disagrees with evidence')
        root_success = out['after_count'] >= req['root_baseline']+req['quantity']
        if row['reward'] != float(root_success):
            raise ValueError('root reward changed by effect contract')
        if root_success:
            completed.add(key)
        expected_context = effect_context(req, out['before_count'])
        actual = [row['phase_context'][data['context_fields'].index(k)] for k in EXTRA_FIELDS]
        if any(not math.isfinite(v) or abs(v-e)>1e-7 for v,e in zip(actual, expected_context)):
            raise ValueError('effect request disconnected from policy input')
        if success:
            peaks[key] = out['after_count']
        if terminal:
            live = None
        previous = row
