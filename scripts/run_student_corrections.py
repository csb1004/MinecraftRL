"""One-use, predeclared train/development student-state correction collection."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from run_rl_batch import ROOT, rearm_allowed, wait_owned
from minecraftrl.runtime import distribution, write_json
from minecraftrl.stall_replay import log_count
from minecraftrl.world_store import digest


def audit(run, report):
    rows = json.loads((run/'correction-rows.json').read_text())
    initial = json.loads((run/'initial-state.json').read_text())['server']
    assert report['execution_valid'] and report['worker_exit_code'] == 0
    assert not report['rl_training_eligible'] and report['learned_policy_loaded']
    assert not report['process_cleanup']['remaining_live_pids']
    release = json.loads((run/'input-release-state.json').read_text())
    assert release['verified'] and release['held_key_count'] == 0
    previous, stopped, accepted, count = initial, False, False, 0
    inference, cycles = defaultdict(list), defaultdict(list)
    frames, owners = [], Counter()
    owner_order = {'student': 0, 'handoff_neutral': 1, 'teacher': 2}
    last_owner = 0
    for line in (run/'events.jsonl').open(encoding='utf-8'):
        event = json.loads(line)
        if event['event'] == 'stop_acknowledged': stopped = True
        if event['event'] == 'target_reached': accepted = True
        if event['event'] in ('policy_dispatch_started', 'inference_started'):
            assert not stopped and not accepted
        if event['event'] == 'cycle_timing':
            cycles[rows[event['step']-1]['owner']].append(event['cycle_ms'])
        if event['event'] != 'transition': continue
        row = rows[count]
        assert event['step'] == count+1 and row['frame'] == f'frames/{count:06d}.png'
        assert row['action'] == event['action'] and row['owner'] == event['controller']
        assert row['server_before'] == previous and row['server_after'] == event['info']['workshop']
        assert row['loss_eligible'] == (row['owner'] == 'teacher')
        assert owner_order[row['owner']] >= last_owner
        last_owner = owner_order[row['owner']]
        assert row['server_after']['server_tick'] - previous['server_tick'] == 1
        frames.append(dict(path=row['frame'], sha256=digest(run/row['frame']), loss_eligible=row['loss_eligible']))
        owners[row['owner']] += 1
        inference[row['owner']].append(event['inference_ms'])
        previous = row['server_after']; count += 1
    assert stopped and count == report['steps'] == len(rows)
    assert owners['handoff_neutral'] <= 1
    assert owners['teacher'] == report['expert_labels']
    gained = log_count(previous)-log_count(initial)
    assert (gained > 0) == report['server_target_reached']
    if owners['teacher']:
        handoff, = report['handoffs']
        assert handoff['verified'] and handoff['teacher_started'] and handoff['held_key_count'] == 0
        assert handoff['step'] == owners['student']+1
        assert handoff['trigger']['step'] == owners['student']
        assert handoff['trigger']['event'] == 'stalled'
        assert rows[handoff['step']-1]['owner'] == 'handoff_neutral'
        assert not report['student_target_reached']
        assert report['teacher_assisted_target_reached'] == (gained > 0)
    return dict(owners=dict(owners), actual_log_delta=gained, frames=frames,
        inference_ms={k: distribution(v) for k,v in inference.items()},
        control_cycle_ms={k: distribution(v) for k,v in cycles.items()},
        post_stop_dispatches=0, released_held_keys=0, remaining_owned_pids=[],
        files_sha256={name:digest(run/name) for name in ('correction-rows.json','events.jsonl','summary.json',
            'initial-state.json','student-prefix.pt','stall-decisions.json')})


def main():
    base = ROOT/'results/student-correction-20261009'
    base.mkdir(exist_ok=False)
    source = json.loads((ROOT/'results/diverse-teacher-worlds-20261008/worlds.json').read_text())['worlds']
    labels = [f'train-{seed}-{kind}' for seed in (104743,104759) for kind in ('far','long-corner')]
    labels += ['development-far','development-corner']
    checkpoint = ROOT/'results/temporal-bc-temporal-20261008/checkpoint.pt'
    embed = ROOT/'results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy'
    assert digest(checkpoint) == '8ce793b4f02af4218eb65d7e741ca3ac021d0caee1d891cc324722eeb288fdb9'
    assert digest(embed) == '810512d05a7f228d77aed45f29a21277156f934408e2387f2ae3fd6f4217be40'
    frozen = {str(p):digest(p) for p in (checkpoint,embed,ROOT/'docs/research-spec.md',
        ROOT/'weights/steve1/steve1.weights', ROOT/'docs/student-correction-protocol-2026-10-09.md')}
    state = dict(completed=False, active=None, runs=[], errors=[], optimizer_updates=0,
        planned_labels=labels, checkpoint=str(checkpoint), frozen=frozen, sampling_seed=9701,
        worlds={k:source[k] for k in labels}, scope='explicit demonstration collection; no policy promotion')
    def save(): write_json(base/'experiment.json',state)
    save()
    try:
        for label in labels:
            world = source[label]
            assert digest(world['path']) == world['sha256']
            rearm_allowed(); state['active'] = label; save()
            log = base/(label+'.log'); report = None
            print('collect '+label,flush=True)
            try:
                wait_owned([sys.executable,str(ROOT/'scripts/run_rl_pilot.py'), '--seed',str(world['seed']),
                    '--split','train' if world['split']=='train' else 'eval', '--sampling-seed','9701',
                    '--student-correction', '--checkpoint',str(checkpoint), '--embed',str(embed),
                    '--world',world['path'], '--reward-mode','wood-potential-v4','--limit-s','180'], log)
            finally:
                lines = log.read_text(encoding='utf-8').splitlines() if log.exists() else []
                if lines:
                    run = Path(lines[0])/'run'
                    if (run/'summary.json').exists():
                        report = json.loads((run/'summary.json').read_text())
                        state['runs'].append(dict(label=label, path=str(run), split=world['split'],
                            seed=world['seed'], report=report, audited=False)); save()
            result = audit(run,report)
            write_json(base/(label+'-audit.json'),result)
            state['runs'][-1].update(audited=True, audit=str(base/(label+'-audit.json')))
            save()
            print(json.dumps(dict(label=label,status=report['status'],owners=result['owners'],
                acquired=result['actual_log_delta'],student_success=report['student_target_reached'])),flush=True)
            assert digest(world['path']) == world['sha256']
        for path, sha in frozen.items(): assert digest(path) == sha
        state.update(completed=True, active=None)
    except BaseException as exc:
        state['errors'].append(repr(exc))
    finally:
        save()
    return 0 if state['completed'] else 2


if __name__ == '__main__': raise SystemExit(main())
