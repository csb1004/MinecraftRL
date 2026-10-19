"""Verify preserved repetitions without launching a game or changing STOP."""
import hashlib
import argparse
import json
from pathlib import Path

from run_local_reliability_batch import ROOT,OUT,sources


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.results_root.resolve()
    if not OUT.is_relative_to(ROOT/'results'):raise ValueError('project-local results required')
    read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    frozen=read(OUT/'verification-freeze.json')
    assert sources()==frozen['source_sha256']
    results=[];world_files=0;jars={}
    for label in ['dev-blind-01','dev-known-01']+[r['label'] for r in frozen['runs']]:
        attempt=OUT/label;result=read(attempt/'reliability.json')
        audit=read(attempt/'run/rm-navigation-audit.json');preserved=read(attempt/'world-preservation.json')
        jar=read(attempt/'run/runtime-jar-cache.json')
        jars[jar['source']]=jar['sha256']
        assert audit['verified'] and audit['action_head_logp_max_error']<=2e-4
        assert audit['log_gain']==result['server_log_gain']
        assert audit['goal_completed']==result['normal_episode_success']
        assert not any(result[k] for k in ('post_stop_actions','held_keys_after_release','remaining_owned_pids'))
        assert audit['actual_llm_calls']==audit['optimizer_updates']==0
        assert all(v>=1 for v in result['saved_log_counts'].values()) if result['normal_episode_success'] else True
        source=attempt/'execution-source'
        recorded=read(attempt/'protocol.json')['source_sha256']
        assert all(sha(source/p)==h for p,h in recorded.items())
        if label.startswith('verify-'):assert recorded==frozen['source_sha256']
        for entry in preserved['files']:
            assert sha(Path(preserved['source'])/entry['path'])==entry['sha256']
            assert sha(Path(preserved['copy'])/entry['path'])==entry['sha256']
            world_files+=1
        results.append(result)
    original=read(ROOT/'results/local-wall-diagnosis-20261017/protocol.json')['frozen_assets']
    assert all(sha(p)==h for p,h in original.items())
    assert sha(ROOT/'docs/research-spec.md')==read(OUT/'protocol.json')['research_spec_sha256']
    assert (ROOT/'results/STOP').is_file()
    assert len(set(jars.values()))==1
    for path,expected in jars.items():
        assert path.startswith('/mnt/c/MinecraftRL/')
        assert sha(ROOT/path[len('/mnt/c/MinecraftRL/'):])==expected
    verified=results[2:];assert len(verified)==6
    output=dict(complete=all(r['normal_episode_success'] for r in verified),
        development_successes=sum(r['normal_episode_success'] for r in results[:2]),development_games=2,
        verification_successes=sum(r['normal_episode_success'] for r in verified),verification_games=6,
        verification_control_s=[r['elapsed_control_s'] for r in verified],
        results=results,source_files_frozen=len(frozen['source_sha256']),
        preserved_world_files=world_files,original_assets_unchanged=True,research_spec_unchanged=True,
        runtime_jar_sha256=list(jars.values())[0],
        stop_latched=True,paid_api_calls=0,optimizer_updates=0,final_test_runs=0,
        scope='Same development maze, two starts, three sampling seeds each; privileged local sensor; not general-world reliability')
    (OUT/'final-verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='results'},indent=2))


if __name__=='__main__':main()
