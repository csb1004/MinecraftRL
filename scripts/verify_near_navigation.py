"""Verify the bounded continuation using preserved trajectory and training receipts."""
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/near-navigation-20261015'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--low-lr',action='store_true');args=parser.parse_args()
    base=ROOT/'results/near-navigation-low-lr-20261015' if args.low_lr else BASE
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    workflow=read(base/'workflow.json');assert workflow['completed'] and not workflow['error']
    training=read(base/'bc/training.json');assert training['completed'] and training['root_unchanged'] and not training['error']
    assert training['method']=='bc_continue' and training['source_sha256']=='e033f1ef7902102569d01459d85cdd80252c141f4e1346bdb40832502e8acaea'
    assert digest(ROOT/'results/navigation-refinement-20261014/ppo/mlp/checkpoint.pt')==training['source_sha256']
    for step in workflow['steps']:
        assert digest(base/'workflow-logs'/(step['label']+'.log'))==step['log_sha256']
    assert len(training['models'])==1
    model=training['models'][0];assert model['kind']=='mlp' and model['reload_error']==0 and model['value_unchanged']
    assert model['learning_rate']==(.00002 if args.low_lr else .0001) and model['epoch_budget']==8 and model['batch_size']==32
    assert model['after_train']['kl']<=.03+1e-6
    assert digest(base/'bc/mlp/checkpoint.pt')==model['checkpoint_sha256']
    for path,sha in training['inputs'].items():assert digest(ROOT/path/'navigation-trajectory.pt')==sha
    labels=['correction-104743','correction-104759']
    if model['actor_weights_changed']:labels+=['evaluate-far','evaluate-corner']
    games=[];initial_comparisons=[]
    for label in labels:
        folder=(BASE if label.startswith('correction-') else base)/label;run=folder/'run';work=read(folder/'work.json');report=read(run/'summary.json')
        audit=read(run/'navigation-audit.json');prob=read(run/'navigation-probability-audit.json')
        assert work['completed'] and not work['error'] and audit['verified'] and prob['verified']
        assert report['root_actor_preserved'] and report['rl_updates']==report['actual_llm_calls']==0
        protocol=read(folder/'protocol.json')
        assert digest(ROOT/protocol['actor'])==protocol['actor_sha256'] and report['elapsed_control_s']<=180
        worlds=read(ROOT/'results/diverse-teacher-worlds-20261008/worlds.json')['worlds']
        assert any(w['sha256']==report['initial_world']['sha256']==digest(Path(w['path'])) for w in worlds.values())
        if label.startswith('evaluate-'):
            assert report['mode_steps']['teacher']==report['mode_steps']['neutral']==0
            assert protocol['sampling_seed']==14131 and all(r['actions']<=100 for r in report['navigation_receipts'])
        assert audit['post_stop_actions']==audit['released_keys']==0 and not audit['remaining_owned_pids']
        for evidence in (audit,prob):
            for name,sha in evidence['files'].items():assert digest(run/name)==sha
        for name,sha in work['frozen'].items():assert digest(ROOT/name)==sha
        if label.startswith('evaluate-'):
            old=ROOT/'results/navigation-refinement-20261014'/label.replace('evaluate-','evaluate-ppo-')/'run'
            with Image.open(old/'frames/000000.png') as im:old_sha=hashlib.sha256(im.tobytes()).hexdigest()
            with Image.open(run/'frames/000000.png') as im:new_sha=hashlib.sha256(im.tobytes()).hexdigest()
            old_condition=read(old/'rl-condition.json');new_condition=read(run/'rl-condition.json')
            initial_comparisons.append(dict(world=label,first_pixels_equal=old_sha==new_sha,
                first_pixels_sha256=new_sha,initial_memory_equal=old_condition['initial_recurrent_sha256']==new_condition['initial_recurrent_sha256'],
                backbone_equal=all(old_condition[k]==new_condition[k] for k in ('backbone_tail_sha256','backbone_temporal_sha256'))))
        games.append(dict(label=label,steps=report['steps'],learned_arrivals=report['learned_arrivals'],
            teacher_arrivals=report['teacher_arrivals'],owners=report['mode_steps'],inference_ms=report['inference_ms'],
            goal_receipts=report['navigation_receipts'],root_inventory_success=report['root_inventory_success'],
            control_cycle_ms=report['control_cycle_ms'],gpu=report['gpu'],device_vram=report['device_vram'],
            logp_error=prob['stored_logp_max_error'],probability_audit_sha256=digest(run/'navigation-probability-audit.json')))
    selection=read(base/'development-selection.json') if (base/'development-selection.json').exists() else None
    rm=None
    if workflow['rm_followup_run']:
        rm=read(base/'game-far/run/rm-navigation-audit.json')
        assert rm['verified'] and rm['root_preserved'] and rm['post_stop_dispatches']==rm['held_keys_after_release']==0 and not rm['remaining_owned_pids']
    label_coverage={}
    for group,paths in [('old_train',list(training['inputs'])[:2]),('new_train',list(training['inputs'])[2:4])]:
        goals=[r['goal'] for path in paths for r in read(ROOT/path/'navigation-rows.json') if r['owner']=='teacher']
        label_coverage[group]=dict(labels=len(goals),within_one_block=sum(g[2]*16<=1 for g in goals),
            beyond_135_degrees=sum(abs(math.degrees(math.atan2(g[0],g[1])))>135 for g in goals))
    result=dict(verified=True,training=training,games=games,selection=selection,rm_followup=rm,
        initial_comparisons=initial_comparisons,teacher_label_coverage=label_coverage,
        api_calls=0,default_promoted=False,final_seeds_used=False,
        scope='Train-only teacher labels, frozen parent/value, reused development tasks; demonstrations are not learned success')
    name='near-navigation-low-lr-verification.json' if args.low_lr else 'near-navigation-verification.json'
    write_json(base/'verification.json',result);write_json(ROOT/'docs/reports/2026-10-15'/name,result)
    print(json.dumps(dict(verified=True,retained_updates=model['retained_updates'],games=len(games),selection=selection)))


if __name__=='__main__':main()
