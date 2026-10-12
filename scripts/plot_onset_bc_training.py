"""Export completed matched-update learning curves and development gates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
out = root/'results/onset-bc-work-20261009'
image = out/'learning-curves.png'
if image.exists():
    raise FileExistsError(image)
reports = {}
sources = {}
for mode, folder in [('uniform', 'onset-bc-uniform-20261009'), ('onset', 'onset-bc-focused-20261009')]:
    base = root/'results'/folder
    report = json.loads((base/'training.json').read_text())
    audit = json.loads((base/'supervision-audit.json').read_text())
    assert report['completed'] and audit['verified']
    assert digest(base/'training.json') == audit['receipts_sha256']['training.json']
    curve = [(0, report['before']['splits']['train']['first'])]
    for step in range(52, 261, 52):
        path = base/f'train-onset-{step:03d}.json'
        row = json.loads(path.read_text())
        curve.append((step, row['splits']['train']['first']))
        sources[path.relative_to(root).as_posix()] = digest(path)
    reports[mode] = dict(report=report, curve=curve)
    for name in ('training.json', 'supervision-audit.json'):
        sources[(base/name).relative_to(root).as_posix()] = digest(base/name)
assert reports['uniform']['report']['before'] == reports['onset']['report']['before']
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
colors = {'uniform': '#75869a', 'onset': '#1769aa'}
for mode, data in reports.items():
    curve = data['curve']
    axes[0].plot([s for s, _ in curve], [m['nll']['mean'] for _, m in curve], marker='o', color=colors[mode], label=mode)
    axes[1].semilogy([s for s, _ in curve], [100*m['p_release_attack_and_turn']['mean'] for _, m in curve],
                     marker='o', color=colors[mode], label=mode)
labels = ['Parent', 'Uniform', 'Onset']
development = [reports['onset']['report']['before']['splits']['development'],
               reports['uniform']['report']['after']['splits']['development'],
               reports['onset']['report']['after']['splits']['development']]
for i, (group, label, color) in enumerate([('first', 'First action', '#c46b29'), ('continuation', 'Continuation', '#238443')]):
    axes[2].bar(np.arange(3)+(i-.5)*.34, [d[group]['nll']['mean'] for d in development], .34, color=color, label=label)
axes[0].set(title='Four training onset states', xlabel='Optimizer updates', ylabel='Mean first-action NLL')
axes[1].set(title='Four training onset states', xlabel='Optimizer updates', ylabel='Release attack + yaw command (%)')
axes[2].set(title='Two development demonstrations', ylabel='Mean NLL', xticks=range(3), xticklabels=labels)
for ax in axes:
    ax.legend(frameon=False)
    ax.grid(alpha=.15, axis='y')
fig.suptitle('Matched BC updates: recorded-trajectory predictions, not gameplay success')
fig.savefig(image, dpi=160)
plt.close(fig)
summary = {}
for mode, data in reports.items():
    r = data['report']
    summary[mode] = dict(updates=r['updates'], elapsed_s=r['elapsed_s'], peak_allocated_mib=r['peak_allocated_mib'],
                         teacher_exposures=r['total_teacher_target_exposures'], onset_exposures=r['onset_exposures'],
                         strict_gate=r['runtime_comparison_eligible'], diagnostic_gate=r['diagnostic_eligible'],
                         train=r['after']['splits']['train'], development=r['after']['splits']['development'],
                         development_first_actions={k:v for k,v in r['after']['first_actions'].items() if k.startswith('development')})
write_json(out/'training-comparison.json', dict(conditions=summary, parent=development[0],
    sources_sha256=sources, image_sha256=digest(image), scope='Same parent, learning rate and update count; different target exposure'))
print(image)
