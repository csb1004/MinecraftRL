"""Plot audited server positions from the paired autonomous diagnostic games."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/onset-bc-comparison-20261009'
image = base/'server-routes-v2.png'
if image.exists():
    raise FileExistsError(image)
state = json.loads((base/'experiment.json').read_text())
audit = json.loads((base/'audit.json').read_text())
assert audit['verified'] and state['completed']
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
sources = {str(p.relative_to(root)): digest(p) for p in (base/'experiment.json', base/'audit.json')}
colors = dict(parent='#73879c', uniform='#c46b29', onset='#176c9c')
for ax, kind in zip(axes, ('long_wall', 'natural')):
    for trial in state['runs']:
        if trial['world']['kind'] != kind:
            continue
        run = Path(trial['path'])
        initial = json.loads((run/'initial-state.json').read_text())
        server = initial['server']
        ox, oz = (16*v for v in initial['initial_world']['chunk']) if kind == 'long_wall' else (server['server_x'], server['server_z'])
        positions = [(server['server_x']-ox, server['server_z']-oz)]
        for line in (run/'events.jsonl').open():
            event = json.loads(line)
            if event['event'] == 'transition':
                s = event['info']['workshop']
                positions.append((s['server_x']-ox, s['server_z']-oz))
        sources[str((run/'events.jsonl').relative_to(root))] = digest(run/'events.jsonl')
        row = next(r for r in audit['rows'] if r['label'] == trial['label'])
        label = '{}: {} log, {:.1f}s'.format(trial['model'], row['inventory_delta'].get('oak_log', 0), row['elapsed_control_s'])
        x, z = zip(*positions)
        ax.plot(x, z, color=colors[trial['model']], label=label, alpha=.8)
        ax.scatter(x[0], z[0], color=colors[trial['model']], marker='o', s=22)
        ax.scatter(x[-1], z[-1], color=colors[trial['model']], marker='x', s=45)
        if kind == 'long_wall' and trial['model'] == 'parent':
            # Geometry coordinates come from the preserved initial world receipt.
            end = initial['initial_world']['curriculum']['wall_end']
            ax.fill_between([7, 8], 1, end+1, color='black', alpha=.15)
    title = kind.replace('_', ' ').title()
    if kind == 'natural' and not state['comparison_valid'].get('natural-onset', False):
        title += '\nOnset initial pixels differ; timings are descriptive'
    ax.set(title=title, xlabel='Server x offset (blocks)', ylabel='Server z offset (blocks)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
fig.suptitle('Autonomous development games: circle=start, cross=end')
fig.savefig(image, dpi=160)
plt.close(fig)
write_json(base/'route-plot-sources-v2.json', dict(sources_sha256=sources, image_sha256=digest(image)))
print(image)
