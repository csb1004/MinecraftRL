"""Plot audited server paths; no environment or policy execution."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root / 'results/masked-correction-comparison-20261009'
image = base / 'long-wall-routes.png'
if image.exists():
    raise FileExistsError(image)
state = json.loads((base / 'experiment.json').read_text())
assert json.loads((base / 'final-verification.json').read_text())['verified']
records = [r for r in state['runs'] if r['world']['kind'] == 'long_wall']
assert len(records) == 2 and state['comparison_valid']['long_wall']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
sources = {}
for axis, record, title in zip(axes, records, ('Parent', 'Masked BC')):
    run = Path(record['path'])
    initial = json.loads((run / 'initial-state.json').read_text())
    world = initial['initial_world']
    ox, oz = (16 * v for v in world['chunk'])
    end = world['curriculum']['wall_end']
    assert not world['curriculum']['mirror_z']
    start = initial['server']
    points = [(start['server_x'] - ox, start['server_z'] - oz)]
    source = run / 'events.jsonl'
    sources[str(source.relative_to(root))] = digest(source)
    for line in source.open():
        event = json.loads(line)
        if event['event'] == 'transition':
            server = event['info']['workshop']
            points.append((server['server_x'] - ox, server['server_z'] - oz))
    axis.add_patch(Rectangle((0, 0), 16, 1, color='#54575b'))
    axis.add_patch(Rectangle((0, 0), 1, 16, color='#54575b'))
    axis.add_patch(Rectangle((7, 1), 1, end, color='#54575b'))
    axis.add_patch(Rectangle((7, end + 1), 1, 1.4, color='#def1c6'))
    axis.add_patch(Rectangle((11, 3), 1, 1, color='#89602d'))
    axis.plot(*zip(*points), color='#1769aa', linewidth=1.7)
    axis.scatter(*points[0], color='#238443', s=45, zorder=3, label='Start')
    axis.scatter(*points[-1], color='#c62828', marker='x', s=65, zorder=3, label='End')
    axis.text(8.3, end + 1.3, 'Open route', fontsize=9)
    axis.text(10.4, 2.2, 'Oak log', fontsize=9)
    axis.set(xlim=(0, 14), ylim=(0, 11), aspect='equal', xlabel='Relative x (blocks)',
             ylabel='Relative z (blocks)', title=f'{title}: no log acquired')
    axis.grid(alpha=.15)
    axis.legend(loc='upper left', frameon=False, fontsize=8)
fig.suptitle('Long wall: actual server paths, one 180-second run per policy', fontsize=12)
fig.savefig(image, dpi=160)
plt.close(fig)
write_json(base / 'route-plot-sources.json', dict(image_sha256=digest(image), sources_sha256=sources,
                                                 source='server transitions; no simulated path'))
print(image)
