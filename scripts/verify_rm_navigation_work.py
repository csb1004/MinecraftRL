"""Assemble completion evidence; never runs or repeats a game."""
import json,re,subprocess
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest
from minecraftrl.waypoint_skill import NAMES,same_failed_route


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/rm-navigation-integration-20261014'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    replay=read(out/'real-trace-replay.json')
    assert replay['verified'];games=[];hashes=dict(replay['source_sha256'])
    for revision in (1,2):
        directory=out/('game-v'+str(revision));work=read(directory/'work.json')
        audit=read(directory/'run/rm-navigation-audit.json')
        assert work['completed'] and not work['error'] and audit['verified']
        assert audit['actual_llm_calls']==audit['optimizer_updates']==0
        assert audit['post_stop_dispatches']==audit['post_stop_inferences']==audit['held_keys_after_release']==0
        assert audit['remaining_owned_pids']==[] and audit['root_preserved']
        hashes.update(work['frozen_assets'])
        for name,sha in audit['source_sha256'].items():
            hashes[(directory/'run'/name).relative_to(root).as_posix()]=sha
        raw=read(directory/'run/goal-recovery-rows.json');duplicate_blocks=0
        for path in (directory/'run').glob('planner-input-*.json'):
            packet=read(path);snapshot=packet['runtime_binding'];obs=raw[snapshot['sequence']-1]['server_after']
            blocked={c['skill_id'] for c in packet['memory']['blocked_methods']}
            previous=[n['receipt'] for plan in packet['previous_plans'] for n in plan['nodes'] if n.get('receipt')]
            for name,cap in packet['execution_registry'].items():
                if name not in NAMES:continue
                if any(same_failed_route(cap,obs,r,world_id=snapshot['world_id'],episode_id=snapshot['episode_id']) for r in previous):
                    assert name in blocked;duplicate_blocks+=1
        games.append(dict(revision=revision,steps=audit['steps'],status=audit['status'],
            planner_calls=audit['planner_calls'],installed_plans=audit['installed_plans'],
            navigation_outcomes=[r['outcome'] for r in audit['navigation_receipts']],
            subgoal_outcomes=[dict(skill_id=r['skill_id'],outcome=r['outcome'],steps=r['committed_steps']) for r in audit['subgoal_receipts']],
            log_gain=audit['log_gain'],goal_completed=audit['goal_completed'],
            action_logp_max_error=audit['action_head_logp_max_error'],post_stop_dispatches=0,
            post_stop_inferences=0,keys_remaining=0,owned_pids_remaining=[],prior_failed_directions_blocked=duplicate_blocks))
    for path,sha in hashes.items():assert digest(root/path)==sha,path
    tests={}
    for platform,name in [('windows','windows-tests-completed-final.log'),('wsl','wsl-tests-completed.log')]:
        text=(out/name).read_text(encoding='utf-8-sig')
        assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',text,re.M) and 'FAILED (' not in text
        tests[platform]=dict(total=int(re.search(r'Ran (\d+) tests',text)[1]),
            skipped=int(re.search(r'OK \(skipped=(\d+)\)',text)[1]) if 'OK (skipped=' in text else 0,
            log_sha256=digest(out/name))
    subprocess.run(['git','diff','--check'],cwd=root,check=True)
    assert (root/'results/STOP').read_text(encoding='utf-8-sig').startswith('RL collection ended')
    assert read(out/'process-check.json')['remaining_project_game_processes']==[]
    source={p.relative_to(root).as_posix():digest(p) for folder in ('src/minecraftrl','scripts','tests')
            for p in (root/folder).glob('*.py')}
    result=dict(verified=True,tests=tests,games=games,
        stored_trace_replay=dict(total_steps=replay['total_steps'],navigation_steps=replay['navigation_steps'],
            outcomes=replay['outcomes'],diagnoses=replay['diagnoses']),
        paid_calls=0,optimizer_updates=0,original_assets_preserved=True,automatic_stop_retained=True,
        scope='Local fixed-plan integration and stored-trace replay; no actual LLM integration effectiveness claim',
        source_sha256=source,frozen_and_original_sha256=hashes)
    write_json(out/'final-verification.json',result)
    small={k:v for k,v in result.items() if k not in ('source_sha256','frozen_and_original_sha256')}
    write_json(root/'docs/reports/2026-10-14/rm-navigation-verification.json',small)
    print(json.dumps(small))


if __name__=='__main__':main()
