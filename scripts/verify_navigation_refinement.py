"""Final local evidence inventory; no games, training, network or promotion."""
import json,re,subprocess,math
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/navigation-refinement-20261014'


def main():
    import torch
    torch.set_num_threads(1)
    workflow=json.loads((BASE/'workflow.json').read_text());assert workflow['completed'] and not workflow['error']
    selected=json.loads((BASE/'ppo-selection.json').read_text())['kind'];verified=[];frames={};conditions=set()
    expected=['correction-train-104743','correction-train-104759','correction-development']
    expected+=['evaluate-'+kind+'-'+world for kind in ('prior','linear','mlp','ppo') for world in ('far','corner')]
    expected+=['ppo-train-104743','ppo-train-104759']
    for label in expected:
        folder=BASE/label;work=json.loads((folder/'work.json').read_text());assert work['completed'] and not work['error']
        run=folder/'run';report=json.loads((run/'summary.json').read_text())
        for check in ('navigation-audit.json','navigation-probability-audit.json'):
            audit=json.loads((run/check).read_text());assert audit['verified']
            for name,sha in audit['files'].items():assert digest(run/name)==sha
        for path,sha in work['frozen'].items():assert digest(ROOT/path)==sha
        protocol=json.loads((folder/'protocol.json').read_text())
        assert digest(ROOT/protocol['actor'])==protocol['actor_sha256']
        assert report['elapsed_control_s']<=180
        release=json.loads((run/'input-release-state.json').read_text())
        assert release['verified'] and release['held_key_count']==0
        if protocol['mode'] in ('evaluate','ppo'):
            assert report['mode_steps']['teacher']==report['mode_steps']['neutral']==0
            assert all(r['actions']<=100 for r in report['navigation_receipts'])
        if protocol['mode']=='evaluate':
            world_kind=label.rsplit('-',1)[1]
            frames.setdefault(world_kind,set()).add(digest(run/'frames/000000.png'))
            condition=json.loads((run/'rl-condition.json').read_text())
            assert condition['sampling_seed']==14131
            conditions.add((condition['initial_recurrent_sha256'],condition['backbone_tail_sha256']))
        world=report['initial_world'];world_paths=json.loads((ROOT/'results/diverse-teacher-worlds-20261008/worlds.json').read_text())['worlds']
        assert any(v['sha256']==world['sha256']==digest(Path(v['path'])) for v in world_paths.values())
        assert report['root_actor_preserved'] and not report['process_cleanup']['remaining_live_pids']
        assert report['actual_llm_calls']==0 and report['rl_updates']==0
        data=torch.load(run/'navigation-trajectory.pt',map_location='cpu',weights_only=False)
        for row in data['rows']:
            assert torch.isfinite(row['features']).all()
            assert all(math.isfinite(row[k]) for k in ('old_logp','value','reward'))
            assert all(math.isfinite(v) for v in row['motion_goal'])
        del data
        verified.append(dict(label=label,steps=report['steps'],learned_arrivals=report['learned_arrivals'],teacher_arrivals=report['teacher_arrivals']))
    assert set(frames)=={'far','corner'} and all(len(hashes)==1 for hashes in frames.values())
    assert len(conditions)==1
    tests={}
    for name in ('windows-tests-final.log','wsl-tests-final.log'):
        text=(BASE/name).read_text(encoding='utf-8-sig')
        assert re.search(r'^OK(?: \(skipped=\d+\))?$',text,re.M) and not re.search(r'^FAILED',text,re.M)
        tests[name]=dict(count=int(re.search(r'Ran (\d+) tests',text).group(1)),sha256=digest(BASE/name))
    assert subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True).returncode==0
    models={}
    for stage,kinds in (('bc',('linear','mlp')),('ppo',(selected,))):
        training=json.loads((BASE/stage/'training.json').read_text());assert training['completed'] and training['root_unchanged']
        for model in training['models']:
            assert model['reload_error']==0
            assert digest(BASE/stage/model['kind']/'checkpoint.pt')==model['checkpoint_sha256']
        models[stage]=dict(report_sha256=digest(BASE/stage/'training.json'),models=training['models'])
    bc=torch.load(BASE/'bc'/selected/'checkpoint.pt',map_location='cpu',weights_only=False)
    ppo=torch.load(BASE/'ppo'/selected/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ppo['source_sha256']==digest(BASE/'bc'/selected/'checkpoint.pt')
    changes={}
    for section in ('actor','navigation_value'):
        assert set(bc[section])==set(ppo[section])
        changed=[k for k in bc[section] if not torch.equal(bc[section][k],ppo[section][k])]
        changes[section]=dict(changed_tensors=len(changed),maximum_absolute_change=max(
            ((bc[section][k]-ppo[section][k]).abs().max().item() for k in changed),default=0.))
    result=dict(verified=True,runs=verified,total_transitions=sum(r['steps'] for r in verified),tests=tests,
        models=models,retained_ppo_changes=changes,all_saved_features_logp_value_finite=True,api_calls=0,root_spec_preserved=True,
        evaluation_first_frames={kind:next(iter(hashes)) for kind,hashes in frames.items()},
        evaluation_initial_recurrent_and_backbone_equal=True,
        scope='local development; frozen known goals; no final 5-seed evaluation or promotion')
    write_json(BASE/'final-verification.json',result)
    small=dict(result);small['models']={stage:dict(report_sha256=value['report_sha256']) for stage,value in models.items()}
    write_json(ROOT/'docs/reports/2026-10-14/navigation-refinement-verification.json',small)
    print(json.dumps(small))


if __name__=='__main__':main()
