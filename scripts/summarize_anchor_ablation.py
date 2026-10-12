"""Compare the preregistered KL ablation with its frozen control; no selection/training."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root=Path(__file__).resolve().parents[1]
out=root/'results/coverage-anchor-ablation-20261009'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
control_path=root/'results/onset-coverage-training-20261009'
reports=[read(path/'training.json') for path in (control_path,out)]
assert read(out/'supervision-audit.json')['verified']
assert all(r['completed'] and r['error'] is None and r['updates']==662 for r in reports)
assert reports[0]['before']==reports[1]['before']
dest=out/'comparison.json'
if dest.exists():raise FileExistsError(dest)
rows=[]
for label,r in zip(('KL 1','KL 0.1'),reports):
    after=r['after'];train=after['splits']['train'];dev=after['splits']['development']
    rows.append(dict(label=label,training_onset_nll=train['first']['nll']['mean'],
        training_onset_turn_probability=train['first']['p_release_attack_and_turn']['mean'],
        development_onset_nll=dev['first']['nll']['mean'],development_continuation_nll=dev['continuation']['nll']['mean'],
        development_turn_probability={k:v['p_release_attack_and_turn'] for k,v in after['first_actions'].items() if k.startswith('development')},
        training_teacher_parent_kl=train['teacher']['reference_kl']['mean'],
        diagnostic_eligible=r['diagnostic_eligible'],strict_gate=r['runtime_comparison_eligible']))
result=dict(verified=True,optimizer_updates_per_condition=662,rows=rows,
    same_parent_and_initial_predictions=True,same_train_targets_and_schedule=True,
    training_prediction_confirmed=(rows[1]['training_onset_nll']<rows[0]['training_onset_nll'] and
                                  rows[1]['training_onset_turn_probability']>rows[0]['training_onset_turn_probability']),
    all_development_turn_probabilities_improved=all(rows[1]['development_turn_probability'][k]>v for k,v in rows[0]['development_turn_probability'].items()),
    source_sha256={str((p/'training.json').relative_to(root)):digest(p/'training.json') for p in (control_path,out)},
    scope='Recorded teacher states. No autonomous navigation/acquisition, PPO update or final evaluation claim.')
write_json(dest,result)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
colors=['#667781','#1769aa']
for p,r,label,color in zip((control_path,out),reports,('KL 1','KL 0.1'),colors):
    curve=[(0,r['before']['splits']['train']['first'])]
    for step in sorted({math.ceil(662*i/5) for i in range(1,6)}):
        curve.append((step,read(p/f'train-onset-{step:03d}.json')['splits']['train']['first']))
    axes[0].plot([s for s,_ in curve],[100*v['p_release_attack_and_turn']['mean'] for _,v in curve],'o-',label=label,color=color)
for i,(row,color) in enumerate(zip(rows,colors)):
    axes[1].bar([j+(i-.5)*.34 for j in range(2)],
        [100*row['development_turn_probability'][k] for k in ('development-far','development-corner')],.34,label=row['label'],color=color)
axes[0].set(title='Training: first release + turn',xlabel='Updates',ylabel='Probability (%)',ylim=(0,100))
axes[1].axhline(.5,color='#b5362a',linestyle='--',label='Diagnostic floor (0.5%)')
axes[1].set(title='Development: first release + turn',xticks=[0,1],xticklabels=['Far','Corner'],yscale='log',ylabel='Probability (%)')
axes[2].bar([r['label'] for r in rows],[r['development_continuation_nll'] for r in rows],color=colors)
axes[2].set(title='Development: teacher continuation',ylabel='Mean NLL')
axes[0].legend(frameon=False);axes[1].legend(frameon=False,fontsize=8)
for ax in axes:ax.grid(alpha=.15,axis='y')
fig.suptitle('Same data, parent, seed and 662 updates | only KL weight changed')
fig.savefig(out/'anchor-comparison.png',dpi=160);plt.close(fig)
print(json.dumps(result))
