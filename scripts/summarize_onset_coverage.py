"""Describe actual teacher handoff poses, independent of intended spawn positions."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


root = Path(__file__).resolve().parents[1]
base = root/'results/onset-coverage-collection-20261009'
out = base/'pose-coverage.json'
if out.exists():
    raise FileExistsError(out)
audit_path = base/'final-audit.json'
audit = json.loads(audit_path.read_text())
assert audit['verified']
old_path = root/'results/onset-bc-work-20261009/onset-state-coverage.json'
old = json.loads(old_path.read_text())['rows']
poses = []
for r in old:
    assert digest(root/r['frame']) == r['frame_sha256']
    if r['split'] != 'train':
        continue
    x, z = r['relative_position']
    mirror = r['wall']['mirror_z']
    yaw = 180-r['yaw'] if mirror else r['yaw']
    poses.append(dict(label=r['session'], source='original', x=x, z=16-z if mirror else z,
                      yaw=(yaw+180)%360-180, pitch=r['pitch'], frame_sha256=r['frame_sha256']))
original = list(poses)
for r in audit['results']:
    if not r['eligible']:
        continue
    p = r['onset_pose']
    closest = min(original, key=lambda a: math.hypot(a['x']-p['x'], a['z']-p['canonical_z']))
    poses.append(dict(label=r['label'], source='new', x=p['x'], z=p['canonical_z'], yaw=p['canonical_yaw'],
        pitch=p['pitch'], frame_sha256=p['frame_sha256'], nearest_original=closest['label'],
        nearest_original_distance_blocks=math.hypot(closest['x']-p['x'], closest['z']-p['canonical_z']),
        yaw_difference_to_nearest=abs((p['canonical_yaw']-closest['yaw']+180)%360-180),
        pitch_difference_to_nearest=abs(p['pitch']-closest['pitch'])))
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for source, color, marker in [('original', '#78899b', 'o'), ('new', '#1476a8', '^')]:
    subset = [p for p in poses if p['source'] == source]
    axes[0].scatter([p['x'] for p in subset], [p['z'] for p in subset], c=color, marker=marker, label=source)
    for p in subset:
        yaw = math.radians(p['yaw'])
        axes[0].arrow(p['x'], p['z'], -.65*math.sin(yaw), .65*math.cos(yaw), color=color, head_width=.13)
    axes[1].scatter([p['yaw'] for p in subset], [p['pitch'] for p in subset], c=color, marker=marker, label=source)
axes[0].set(xlabel='Canonical x (blocks)', ylabel='Canonical z (blocks)', title='Actual first teacher poses', xlim=(0, 16), ylim=(0, 16))
axes[0].set_aspect('equal')
axes[1].set(xlabel='Canonical yaw (degrees)', ylabel='Pitch (degrees)', title='Actual viewing directions')
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
fig.suptitle('Training states only; mirrored worlds normalized; no claim of broad coverage')
image = base/'pose-coverage.png'
fig.savefig(image, dpi=160)
plt.close(fig)
write_json(out, dict(poses=poses, original_states=len(original), new_teacher_states=len(poses)-len(original),
    exact_frame_hash_duplicates=len(poses)-len({p['frame_sha256'] for p in poses}),
    totals=audit['totals'], source_sha256={str(p.relative_to(root)): digest(p) for p in (audit_path, old_path)},
    image_sha256=digest(image), scope='Descriptive actual training poses; not a generalization metric'))
print(json.dumps(dict(original_states=len(original), new_teacher_states=len(poses)-len(original))))
