"""Linear probe: can the crosshair block be read from the frozen STEVE-1 CNN features? (task E 2026-11-07)

Usage (WSL steve1 env): python scripts/probe_cnn_blocks.py --train RUN_DIR... --test RUN_DIR... --out NEW_DIR
       [--extra-test NAME=RUN_DIR,RUN_DIR...] [--train-per-class 1500] [--test-per-class 600] [--seed 7]
Labels are the server's crosshair block (privileged, analysis only). Frames are resized to 128x128 exactly as the
agent does. Features: the policy's image encoder output (img_preprocess + img_process, frozen official STEVE-1
weights), and two pixel baselines: the 16x16 crosshair patch and the whole image at 16x16. A softmax regression is
fitted on train frames only and scored on test frames. No game, gameplay, optimizer step on the policy, or API.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from minecraftrl.pipeline.frame_labels import CLASSES, balanced_sample, labelled_frames  # noqa: E402


HUD_ROWS = 20   # the bottom rows of the 128x128 image hold the hearts and the hotbar


def load_images(items, mask_hud=False):
    import cv2
    from PIL import Image
    out = np.empty((len(items), 128, 128, 3), dtype=np.uint8)
    for i, item in enumerate(items):
        out[i] = cv2.resize(np.asarray(Image.open(item['path']).convert('RGB')), (128, 128), interpolation=cv2.INTER_LINEAR)
    if mask_hud: out[:, -HUD_ROWS:] = 0
    return out


def image_encoder(device):
    import torch
    sys.path.insert(0, str(ROOT / 'external/steve1'))
    from gym3.types import DictType
    from steve1.VPT.agent import CameraHierarchicalMapping
    from steve1.embed_conditioned_policy import MinecraftAgentPolicy
    with open(ROOT / 'weights/steve1/2x.model', 'rb') as stream: parameters = pickle.load(stream)
    kwargs = parameters['model']['args']['net']['args']
    head = dict(parameters['model']['args']['pi_head_opts']); head['temperature'] = float(head['temperature'])
    space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())
    policy = MinecraftAgentPolicy(policy_kwargs=kwargs, pi_head_kwargs=head, action_space=space)
    policy.load_state_dict(torch.load(str(ROOT / 'weights/steve1/steve1.weights'), map_location='cpu'), strict=True)
    pre, enc = policy.net.img_preprocess.to(device).eval(), policy.net.img_process.to(device).eval()
    del policy

    def encode(images, batch=128):
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.from_numpy(images[i:i + batch]).to(device)[:, None]          # (B, 1, 128, 128, 3)
                feats.append(enc(pre(x))[:, 0].float().cpu().numpy())
        return np.concatenate(feats)
    return encode


def pixel_features(images):
    import cv2
    centre = images[:, 56:72, 56:72].reshape(len(images), -1).astype(np.float32) / 255.
    small = np.stack([cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA) for im in images]).reshape(len(images), -1)
    return centre, small.astype(np.float32) / 255.


def fit_softmax(x, y, n_classes, *, weight_decay=1e-3, steps=300, seed=0):
    import torch
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std).float(); yt = torch.from_numpy(y).long()
    layer = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.LBFGS(layer.parameters(), lr=1., max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(layer(xt), yt) + weight_decay * (layer.weight ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)

    def predict(z):
        with torch.no_grad(): return layer(torch.from_numpy((z - mean) / std).float()).argmax(1).numpy()
    return predict


def score(pred, y, labels):
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y, pred): confusion[t, p] += 1
    recall = {labels[i]: (confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None for i in range(len(labels))}
    present = [v for v in recall.values() if v is not None]
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(np.mean(present)),
                recall={k: None if v is None else float(v) for k, v in recall.items()},
                confusion=confusion.tolist(), n=int(len(y)))


def collect(dirs, per_class, seed):
    items = []
    for d in dirs:
        d = Path(d)
        if (d / 'frames').is_dir() and (d / 'telemetry.jsonl').exists(): items += labelled_frames(d)
    return balanced_sample(items, per_class, seed), len(items)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--train', nargs='+', type=Path, required=True)
    p.add_argument('--test', nargs='+', type=Path, required=True)
    p.add_argument('--extra-test', action='append', default=[], help='NAME=DIR,DIR (another scored set)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-per-class', type=int, default=1500)
    p.add_argument('--test-per-class', type=int, default=600)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--mask-hud', action='store_true', help='black out the bottom HUD rows (held-item cues)')
    a = p.parse_args(argv)
    if a.out.exists():
        print(f'rejected: output folder exists: {a.out}', file=sys.stderr); return 2
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sets = {'train': collect(a.train, a.train_per_class, a.seed), 'test': collect(a.test, a.test_per_class, a.seed)}
    for spec in a.extra_test:
        name, dirs = spec.split('=', 1)
        sets[name] = collect([Path(x) for x in dirs.split(',')], a.test_per_class, a.seed)
    labels = [c for c in CLASSES if any(i['label'] == c for i in sets['train'][0])]
    index = {c: i for i, c in enumerate(labels)}
    encode = image_encoder(device)
    data = {}
    for name, (items, available) in sets.items():
        items = [i for i in items if i['label'] in index]
        images = load_images(items, a.mask_hud)
        centre, small = pixel_features(images)
        data[name] = dict(y=np.array([index[i['label']] for i in items]), cnn=encode(images), centre=centre, small=small,
                          available=available, counts={c: sum(i['label'] == c for i in items) for c in labels})
    results = {}
    for feature in ('cnn', 'centre', 'small'):
        predict = fit_softmax(data['train'][feature], data['train']['y'], len(labels), seed=a.seed)
        results[feature] = {name: score(predict(d[feature]), d['y'], labels) for name, d in data.items()}
    majority = int(np.bincount(data['train']['y']).argmax())
    results['majority'] = {name: score(np.full(len(d['y']), majority), d['y'], labels) for name, d in data.items()}
    a.out.mkdir(parents=True)
    report = dict(labels=labels, device=device, feature_dims={k: int(data['train'][k].shape[1]) for k in ('cnn', 'centre', 'small')},
                  sets={name: dict(counts=d['counts'], frames_available=d['available']) for name, d in data.items()},
                  dirs=dict(train=list(map(str, a.train)), test=list(map(str, a.test)), extra=a.extra_test),
                  seed=a.seed, mask_hud=a.mask_hud, hud_rows=HUD_ROWS if a.mask_hud else 0, results=results,
                  peak_cuda_mib=torch.cuda.max_memory_allocated() / 2 ** 20 if device == 'cuda' else 0.,
                  scope='frozen official STEVE-1 image encoder; probe fitted on train frames only; no policy update')
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# CNN 블록 인식 선형 probe', '', '| 특징 | 집합 | 정확도 | 균형 정확도 | ' + ' | '.join(labels) + ' |',
             '|---|---|---|---|' + '---|' * len(labels)]
    for feature, per_set in results.items():
        for name, s in per_set.items():
            rec = ' | '.join('-' if s['recall'][c] is None else f"{s['recall'][c]:.2f}" for c in labels)
            lines.append(f"| {feature} | {name} | {s['accuracy']:.2f} | {s['balanced_accuracy']:.2f} | {rec} |")
    (a.out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
