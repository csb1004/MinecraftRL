"""Verify a complete correction cache against actual collected student inputs."""
import argparse
import json
from pathlib import Path

import torch

from minecraftrl.rl_policy import state_fingerprint
from minecraftrl.runtime import write_json
from minecraftrl.temporal_bc import episode_rows
from minecraftrl.world_store import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.features.resolve()
    if not path.is_relative_to(root/'results'):
        raise ValueError('project-local feature cache required')
    out = path.parent/'source-replay-audit.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    receipt = json.loads((path.parent/'features.json').read_text())
    assert receipt['completed'] and digest(path) == receipt['features_sha256']
    data = torch.load(path, map_location='cpu', weights_only=False)
    groups = episode_rows(data)
    reports, labels, context = [], 0, 0
    for s in data['plan']['sessions']:
        run = root/s['run']
        prefix = run/'student-prefix.pt'
        assert digest(prefix) == s['files_sha256']['student-prefix.pt']
        original = torch.load(prefix, map_location='cpu', weights_only=False)
        for feature_key, original_key in (('tail_initial', 'backbone_tail'), ('temporal_initial', 'backbone_temporal')):
            assert state_fingerprint(data[feature_key]) == state_fingerprint(original['initial_heads'][original_key])
        assert state_fingerprint(data['temporal_initial_memory']) == state_fingerprint(original['initial_recurrent_state'][-1])
        rows, student = groups[s['id']], original['rows']
        assert len(student) == s['handoffs'][0]['step']-1
        error = max(float((a['features']-b['features']).abs().max()) for a, b in zip(student, rows))
        assert error < 1e-5, (s['id'], error)
        for i, r in enumerate(rows):
            assert r['loss_eligible'] == s['frames'][i]['loss_eligible']
            assert (r['action'] is not None) == r['loss_eligible']
            labels += r['loss_eligible']
            context += not r['loss_eligible']
        reports.append(dict(session=s['id'], original_student_inputs=len(student), max_feature_error=error,
                            expert_targets=sum(r['loss_eligible'] for r in rows)))
        del original
    assert labels == sum(s['expert_labels'] for s in data['plan']['sessions'])
    assert labels+context == len(data['rows']) and digest(path) == receipt['features_sha256']
    report = dict(verified=True, features_sha256=digest(path),
                  original_student_inputs=sum(r['original_student_inputs'] for r in reports),
                  source_replays=reports, expert_targets=labels, context_without_targets=context,
                  game_reruns=0, source_hash_preserved=True)
    write_json(out, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
