"""Export the fixed development phase/onset measurements, without model execution."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root / 'results/correction-action-diagnosis-20261009'
report = json.loads((base / 'report.json').read_text())
assert report['completed']
source = base / 'metrics.json'
assert digest(source) == report['results_sha256']['metrics.json']
rows = json.loads(source.read_text())
image = base / 'development-diagnosis.png'
onset_image = base / 'onset-by-frame.png'
if image.exists() or onset_image.exists():raise FileExistsError('diagnosis figures already exist')

def metric(model, kind, group, key):
    found = [r for r in rows if r['model'] == model and r['split'] == 'development' and
             r['session'] == 'all' and r['kind'] == kind and r['group'] == group]
    assert len(found) == 1
    return found[0]['metrics'][key]['mean']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
phases = ['reposition', 'gap_approach', 'gap_crossing', 'reapproach', 'harvest', 'clear_stump', 'collect']
labels = ['Reposition', 'Approach gap', 'Cross gap', 'Return to log', 'Harvest', 'Clear stump', 'Collect']
commands = ['p_attack', 'p_move', 'p_camera_on', 'p_target_yaw_sign']
command_labels = ['Attack', 'Movement', 'Camera enabled', 'Teacher yaw sign\n(including gate)']
for i, (model, color) in enumerate([('parent','#75869a'),('candidate','#1769aa')]):
    axes[0].barh(np.arange(len(phases))+(i-.5)*.36,
                 [metric(model,'leg',phase,'nll') for phase in phases], height=.36, color=color, label=model)
    axes[1].bar(np.arange(len(commands))+(i-.5)*.36,
                [100*metric(model,'onset','first8',key) for key in commands], width=.36, color=color, label=model)
axes[0].set(yticks=np.arange(len(phases)), yticklabels=labels, xlabel='Mean teacher-label NLL (lower is better)',
            title='Recorded development trajectories')
axes[0].invert_yaxis()
axes[1].set(xticks=np.arange(len(commands)), xticklabels=command_labels, ylabel='Command probability (%)',
            title='First 8 teacher frames per development episode', ylim=(0,100))
axes[1].tick_params(axis='x', labelsize=9)
for axis in axes:
    axis.legend(frameon=False)
    axis.grid(axis='x' if axis is axes[0] else 'y', alpha=.15)
fig.suptitle('Correction diagnosis: teacher-state predictions, not gameplay success', fontsize=12)
fig.savefig(image, dpi=160)
plt.close(fig)
predictions = base / 'demonstration-predictions.jsonl'
assert digest(predictions) == report['results_sha256'][predictions.name]
onset, first = {}, []
seen = set()
with predictions.open() as stream:
    for line in stream:
        row = json.loads(line)
        key = (row['model'], row['session'], row['frame'])
        assert key not in seen
        seen.add(key)
        index = row['label']['teacher_index']
        if index == 0:first.append(row)
        if row['split'] == 'development' and index is not None and index < 8:
            onset.setdefault((row['model'],row['session']), []).append(row)
assert len(seen) == 9922 and len(first) == 12
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
for axis, session in zip(axes, ('development-far','development-corner')):
    for model, color in [('parent','#75869a'),('candidate','#1769aa')]:
        sequence = sorted(onset[model,session], key=lambda r:r['label']['teacher_index'])
        assert [r['label']['teacher_index'] for r in sequence] == list(range(8))
        axis.semilogy(range(8),[100*r['metrics']['p_target_yaw_sign'] for r in sequence],
                      marker='o',color=color,label=model)
    axis.set(title=session, xlabel='Prior teacher actions in the recorded trajectory',
             ylabel='Teacher yaw-sign command probability (%)', ylim=(.0001,110), xticks=range(8))
    axis.grid(alpha=.2, which='both')
    axis.legend(frameon=False)
fig.suptitle('Starting recovery vs following teacher-altered observations',fontsize=12)
fig.savefig(onset_image,dpi=160)
plt.close(fig)
contributions = []
for phase in phases:
    r = next(r for r in rows if r['model']=='candidate' and r['split']=='development' and
             r['session']=='all' and r['kind']=='leg' and r['group']==phase)
    delta = metric('parent','leg',phase,'nll') - metric('candidate','leg',phase,'nll')
    contributions.append(dict(phase=phase,frames=r['frames'],mean_nll_reduction=delta,
                              total_nll_reduction=delta*r['frames']))
total = sum(r['total_nll_reduction'] for r in contributions)
write_json(base/'first-action-detail.json',dict(first_teacher_predictions=first,loss_contributions=contributions,
    crossing_and_reapproach_share=sum(r['total_nll_reduction'] for r in contributions
                                     if r['phase'] in ('gap_crossing','reapproach'))/total,
    first_teacher_train_labels=4,total_teacher_train_labels=1005,
    source_sha256=digest(predictions),scope='Frame-level inspection within the preregistered first-eight window'))
write_json(base/'plot-sources.json',dict(metrics_sha256=digest(source),predictions_sha256=digest(predictions),
    image_sha256={image.name:digest(image),onset_image.name:digest(onset_image)},script_sha256=digest(Path(__file__))))
print(image)
print(onset_image)
