"""Prepare the exact bounded live LLM validation scope; no network or gameplay."""
import argparse
import json
from pathlib import Path

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def prepare(output,run,probe):
    root=Path(__file__).resolve().parents[1];output=output.resolve();run=run.resolve();probe=probe.resolve()
    if not all(p.is_relative_to(root/'results') for p in (output,run,probe)):raise ValueError('project results required')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    audit=read(run/'subgoal-audit.json');effect=read(probe/'report.json')
    if not audit['verified'] or audit['runtime_error'] or not audit['subgoal_receipts'] or not effect['completed']:
        raise ValueError('reviewed real subgoal execution and condition probe required')
    assets=dict(checkpoint='results/onset-coverage-training-20261009/checkpoint.pt',
        embed='results/rl-train-104729-20261005T022109534595Z/run/goal-embed.npy',
        recovery_embed='results/goal-conditioning-diagnosis-20261009/explore_9711.npy',
        world='results/diverse-teacher-worlds-20261008/development-corner.zip',
        skill_catalogue='results/subgoal-skills-20261010/embeddings/catalogue.json')
    sources=list((root/'src/minecraftrl').glob('*.py'))
    sources += [root/'scripts'/n for n in ('run_goal_recovery_pilot.py','run_recovery_broker.py',
        'run_authorized_subgoal_live.py','audit_fresh_target_handoff.py','audit_subgoal_schedule.py')]
    inputs=[root/p for p in assets.values()]+[run/'subgoal-audit.json',probe/'report.json',root/'docs/research-spec.md']
    manifest=dict(prepared=True,paid_authorization_remaining=0,maximum_calls=1,model='gpt-5.3-codex',
        max_output_tokens=2048,timeout_s=20,retries=0,control_limit_s=180,world_seed=104761,sampling_seed=9821,
        maximum_games=1,optimizer_updates=0,assets=assets,previous_fixed_run=str(run),
        source_sha256={p.relative_to(root).as_posix():digest(p) for p in sources+inputs},
        scope='One fresh development game; one LLM subgoal plan after real stall; no second call or automatic retry')
    output.mkdir(exist_ok=False);write_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('run',type=Path);p.add_argument('probe',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.output,a.run,a.probe)))
