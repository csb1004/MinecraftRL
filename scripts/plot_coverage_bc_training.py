"""Plot audited coverage learning and separate development admission metrics."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/onset-coverage-training-20261009'
read = lambda p: json.loads(p.read_text())
report, audit = read(base/'training.json'), read(base/'supervision-audit.json')
assert report['completed'] and audit['verified']
assert digest(base/'training.json') == audit['receipts_sha256']['training.json']
image = base/'learning-and-admission.png'
if image.exists():
    raise FileExistsError(image)
sources = {name:digest(base/name) for name in ('training.json', 'supervision-audit.json')}
curve = [(0, report['before']['splits']['train']['first'])]
for step in sorted({math.ceil(report['planned_updates']*i/5) for i in range(1, 6)}):
    path = base/f'train-onset-{step:03d}.json'
    curve.append((step, read(path)['splits']['train']['first']))
    sources[path.name] = digest(path)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
axes[0, 0].plot([s for s,_ in curve], [m['nll']['mean'] for _,m in curve], 'o-', color='#1769aa')
axes[0, 0].set(title='Training onset states (11)', xlabel='Optimizer updates', ylabel='Mean first-action NLL')
axes[0, 1].plot([s for s,_ in curve], [100*m['p_release_attack_and_turn']['mean'] for _,m in curve], 'o-', color='#238443')
axes[0, 1].set(title='Training onset states (11)', xlabel='Optimizer updates', ylabel='Release attack + yaw (%)')
names = ['development-far', 'development-corner']
for i, (phase, label, color) in enumerate([('before','Parent','#75869a'), ('after','Coverage','#1769aa')]):
    values = [100*report[phase]['first_actions'][name]['p_release_attack_and_turn'] for name in names]
    axes[1, 0].bar([j+(i-.5)*.34 for j in range(2)], values, .34, label=label, color=color)
axes[1, 0].axhline(.5, color='#b5362a', linestyle='--', label='Each-state diagnostic floor (0.5%)')
axes[1, 0].set(title='Untrained development onset states', xticks=[0,1], xticklabels=['Far','Corner'],
               yscale='log', ylabel='Release attack + yaw (%)')
axes[1, 0].legend(frameon=False, fontsize=8)
values = [report[p]['splits']['development']['continuation']['nll']['mean'] for p in ('before','after')]
axes[1, 1].bar(['Parent','Coverage'], values, color=['#75869a','#1769aa'])
axes[1, 1].axhline(values[0]*1.1, color='#b5362a', linestyle='--', label='Maximum admitted NLL')
axes[1, 1].set(title='Development teacher continuation', ylabel='Mean NLL')
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.15, axis='y')
fig.suptitle('Expanded coverage BC | recorded observations, not autonomous gameplay success')
fig.savefig(image, dpi=160)
plt.close(fig)
write_json(base/'plot-sources.json', dict(sources_sha256=sources, image_sha256=digest(image),
           diagnostic_admitted=report['diagnostic_eligible'], strict_gate=report['runtime_comparison_eligible']))
print(image)
