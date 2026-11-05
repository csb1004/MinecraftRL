"""Render one demo run as an mp4 in the replay layout (spec 5.11) for sharing where the HTML replay cannot be opened.

Usage (WSL project environment, needs Pillow and the imageio-ffmpeg binary):
  python scripts/render_demo_video.py RUN_DIR OUT.mp4 [--speed 4] [--title TEXT]
The clock is the recorded time `t`; a hold (game frozen while SQLite/LLM run) shows the last frame with its badge.
Only Malgun Gothic Bold is used: the regular weight misdraws some Hangul (e.g. 갇, 깊) at 20-29 px in Pillow.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minecraftrl.pipeline.viewer import badge_at, events_at, kb_at, read_jsonl, rm_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / '.runtime/steve1/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.1'
BOLD = '/mnt/c/Windows/Fonts/malgunbd.ttf'
W, H, FPS = 1280, 720, 25
VIEW = (16, 64, 784, 496)  # agent view box (16:9)
BG, PANEL, LINE, INK, MUTED = (20, 20, 23), (32, 32, 37), (58, 58, 66), (236, 236, 241), (160, 160, 170)
COL = dict(anomaly=(251, 146, 60), knowledge=(196, 181, 253), replan=(103, 232, 249), success=(74, 222, 128), risk=(248, 113, 113),
           info=(160, 160, 170))
STATUS = {'proposed': ('제안', MUTED), 'supported': ('지지', (74, 222, 128)), 'disputed': ('논쟁', (251, 146, 60)),
          'retracted': ('철회', (248, 113, 113))}
RMS = {'done': ('완료', (74, 222, 128)), 'active': ('진행', (122, 167, 255)), 'held': ('정지', (251, 146, 60)),
       'failed': ('실패', (248, 113, 113)), 'pending': ('대기', MUTED)}
CTL = {'policy': '정책 (STEVE-1)', 'rule': '규칙 스킬', 'gui_close': '세션: GUI 닫기', 'prepare': '준비', 'hold': '정지'}


def font(size): return ImageFont.truetype(BOLD, size)


F_TITLE, F_H, F_TEXT, F_SMALL = font(24), font(17), font(16), font(14)


def fit(draw, text, f, width):
    if draw.textlength(text, font=f) <= width: return text
    while text and draw.textlength(text + '…', font=f) > width: text = text[:-1]
    return text + '…'


def cond(c):
    if not c: return '항상'
    parts = []
    for k, v in c.items():
        parts.append(k + ' ' + (' '.join(('≥' if o == 'gte' else '≤') + str(x) for o, x in v.items()) if isinstance(v, dict) else str(v)))
    return ', '.join(parts)


def claim_text(c):
    obj = c['object']
    if isinstance(obj, dict): obj = ' '.join('%s=%s' % (k, v) for k, v in obj.items())
    return '%s %s %s' % (c['subject'], c['relation'], obj)


def changed_at(timeline, t):
    """{claim id: time of its last change (or first appearance)} over the KB snapshots up to t."""
    seen, when = {}, {}
    for e in timeline:
        if e['t'] > t: break
        if e['kind'] != 'kb_snapshot': continue
        for c in e['claims']:
            if seen.get(c['id']) != c: seen[c['id']] = c; when[c['id']] = e['t'] if e.get('reason') != 'session_start' else -1.
    return when


def render(frame, sample, t, timeline, *, t0, t1, title, run):
    img = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(img)
    d.text((16, 16), title, font=F_TITLE, fill=INK)
    tag = '준비된 시연 월드 · ' + run
    d.text((W - 16 - d.textlength(tag, font=F_SMALL), 22), tag, font=F_SMALL, fill=(122, 167, 255))
    x0, y0, x1, y1 = VIEW
    if frame is not None:
        img.paste(Image.open(frame).convert('RGB').resize((x1 - x0, y1 - y0)), (x0, y0))
    if sample:
        label = CTL.get(sample.get('controller'), sample.get('controller'))
        d.rounded_rectangle((x0 + 8, y0 + 8, x0 + 20 + d.textlength(label, font=F_H), y0 + 36), 6, fill=(0, 0, 0))
        d.text((x0 + 14, y0 + 11), label, font=F_H, fill=INK)
        inv = [('%s ×%d' % (k, v), k == sample.get('server_selected_item')) for k, v in (sample.get('inventory') or {}).items() if v]
        x = x0 + 8; y = y1 - 30
        for text, sel in inv:
            w = d.textlength(text, font=F_SMALL) + 12
            if x + w > x1 - 8: break
            d.rounded_rectangle((x, y, x + w, y + 22), 5, fill=(0, 0, 0), outline=INK if sel else None, width=2)
            d.text((x + 6, y + 3), text, font=F_SMALL, fill=INK); x += w + 6
    badge = badge_at(timeline, t)
    if badge:
        # The font has no ⏸/⛔ glyphs: draw the icon, then the text without it.
        text = badge['text'].lstrip('⏸⛔ ').strip(); w = d.textlength(text, font=F_H) + 52; cx = (x0 + x1) / 2
        d.rounded_rectangle((cx - w / 2, y0 + 44, cx + w / 2, y0 + 78), 8, fill=(194, 65, 12) if badge['kind'] == 'hold' else (185, 28, 28))
        ix, iy = cx - w / 2 + 14, y0 + 52
        if badge['kind'] == 'hold':
            d.rectangle((ix, iy, ix + 5, iy + 18), fill=(255, 255, 255)); d.rectangle((ix + 10, iy, ix + 15, iy + 18), fill=(255, 255, 255))
        else:
            d.ellipse((ix - 1, iy - 1, ix + 17, iy + 17), fill=(255, 255, 255)); d.rectangle((ix + 3, iy + 6, ix + 13, iy + 10), fill=(185, 28, 28))
        d.text((ix + 24, y0 + 50), text, font=F_H, fill=(255, 255, 255))
    # right column: RM progress, what just happened
    rx, rw = 800, W - 816
    d.rounded_rectangle((rx, 64, rx + rw, 250), 8, fill=PANEL); d.text((rx + 12, 72), 'RM 진행', font=F_H, fill=MUTED)
    y = 98
    for s in rm_at(timeline, t)[:7]:
        name, colour = RMS[s['status']]
        d.text((rx + 12, y), fit(d, '%s (%s ×%d)' % (s['id'], s['target'], s['count']), F_TEXT, rw - 90), font=F_TEXT, fill=INK)
        d.text((rx + rw - 12 - d.textlength(name, font=F_TEXT), y), name, font=F_TEXT, fill=colour); y += 21
    d.rounded_rectangle((rx, 258, rx + rw, 496), 8, fill=PANEL); d.text((rx + 12, 266), '방금 일어난 일', font=F_H, fill=MUTED)
    y = 292
    for e in events_at(timeline, t, n=9):
        d.text((rx + 12, y), '%5.1fs' % (e['t'] - t0), font=F_SMALL, fill=MUTED)
        d.text((rx + 66, y), fit(d, e['text'], F_SMALL, rw - 78), font=F_SMALL, fill=COL.get(e['category'], INK)); y += 22
    # knowledge table
    d.rounded_rectangle((16, 504, W - 16, 664), 8, fill=PANEL)
    d.text((28, 510), '지식 (SQLite)', font=F_H, fill=MUTED)
    cols = (28, 560, 860, 960)
    for x, head in zip(cols, ('주장', '조건', '상태', '근거')): d.text((x, 532), head, font=F_SMALL, fill=MUTED)
    kb = kb_at(timeline, t); y = 552
    touched = changed_at(timeline, t)
    rows = sorted(kb['claims'], key=lambda c: (-touched.get(c['id'], -1.), c['id']))  # most recently changed first
    for c in rows[:5]:
        if c['id'] in kb['changed']: d.rectangle((20, y - 2, W - 20, y + 20), fill=(61, 52, 18))
        name, colour = STATUS.get(c['status'], (c['status'], INK))
        d.text((cols[0], y), fit(d, claim_text(c), F_SMALL, cols[1] - cols[0] - 10), font=F_SMALL, fill=INK)
        d.text((cols[1], y), fit(d, cond(c['conditions']), F_SMALL, cols[2] - cols[1] - 10), font=F_SMALL, fill=INK)
        d.text((cols[2], y), name, font=F_SMALL, fill=colour)
        src = (c.get('sources') or ['-'])[0].replace('https://', '')
        obs = c.get('observed') or {}
        # ASCII hyphen: the font has no U+2212 minus.
        if obs.get('support') or obs.get('refute'): src = '게임 관측 +%d/-%d · %s' % (obs.get('support', 0), obs.get('refute', 0), src)
        d.text((cols[3], y), fit(d, src, F_SMALL, W - 28 - cols[3]), font=F_SMALL, fill=MUTED); y += 22
    if len(kb['claims']) > 5: d.text((W - 120, 510), '외 %d개' % (len(kb['claims']) - 5), font=F_SMALL, fill=MUTED)
    if not kb['claims']: d.text((cols[0], y), '비어 있음', font=F_SMALL, fill=MUTED)
    # timeline bar
    d.rounded_rectangle((16, 676, W - 16, 704), 6, fill=PANEL)
    span = max(t1 - t0, 1e-6)
    for e in events_at(timeline, t1, n=10 ** 9):
        if e['category'] in ('info',): continue
        x = 20 + (e['t'] - t0) / span * (W - 40); d.rectangle((x, 680, x + 2, 700), fill=COL[e['category']])
    x = 20 + (t - t0) / span * (W - 40); d.rectangle((x - 1, 672, x + 1, 708), fill=INK)
    clock = '%.1f / %.1f 초' % (t - t0, t1 - t0)
    d.text((W - 16 - d.textlength(clock, font=F_SMALL), 46), clock, font=F_SMALL, fill=MUTED)
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--speed', type=float, default=4.); p.add_argument('--title', default='철 캐줘 — 시연 재생')
    a = p.parse_args(argv)
    timeline = read_jsonl(a.run_dir / 'timeline.jsonl'); samples = {r['step']: r for r in read_jsonl(a.run_dir / 'telemetry.jsonl')}
    frames = [(samples[int(f.stem)]['t'], f, samples[int(f.stem)]) for f in sorted((a.run_dir / 'frames').glob('*.jpg'))
              if int(f.stem) in samples]
    if not frames: p.error('no frames with telemetry in ' + str(a.run_dir))
    t0 = frames[0][0]; t1 = max(frames[-1][0], max(e['t'] for e in timeline)) + 2.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(FFMPEG), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H),
                             '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(a.out)],
                            stdin=subprocess.PIPE)
    i, t, n = 0, t0, 0
    while t <= t1:
        while i + 1 < len(frames) and frames[i + 1][0] <= t: i += 1
        proc.stdin.write(render(frames[i][1], frames[i][2], t, timeline, t0=t0, t1=t1, title=a.title, run=a.run_dir.name).tobytes())
        t += a.speed / FPS; n += 1
    proc.stdin.close(); proc.wait()
    print('%s: %d frames, %.1f s video for %.1f s recorded (x%g)' % (a.out, n, n / FPS, t1 - t0, a.speed))


if __name__ == '__main__':
    main()
