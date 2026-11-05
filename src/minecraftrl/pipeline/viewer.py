"""Offline replay of one demo run (spec 5.11, user-confirmed layout: A + conditions column).

Left: the agent view with the controller badge, inventory and hold/mask badges. Right: RM progress, what just happened,
and the knowledge table (claim / condition / status / evidence; rows changed in the last seconds are highlighted).
Bottom: a timeline with risk, anomaly, knowledge, replan and success marks.

The time axis is the recorded clock, not the step count: a hold freezes the game while the slow path (SQLite, LLM)
runs, and that wait is part of what the replay shows. Nothing is fetched from the network.
"""
import html
import json
import os
import shutil
from pathlib import Path

FRESH_S = 3.  # how long a changed knowledge row stays highlighted
MASK_S = 1.   # how long a masked-attack badge stays visible
CONTROLLER_LABELS = {'policy': '정책 (STEVE-1)', 'rule': '규칙 스킬', 'gui_close': '세션: GUI 닫기', 'prepare': '준비', 'hold': '정지'}
ROUTES = {'replan_from_kb': 'SQLite 적중', 'acquire_then_replan': 'SQLite 없음 → LLM', 'retry_stage': '같은 단계 재시도',
          'retry_limit': '재시도 상한', 'ask_llm': '결정 미구현'}
STATUS_LABELS = {'proposed': '제안', 'supported': '지지', 'disputed': '논쟁', 'retracted': '철회'}


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def kb_at(timeline, t, fresh=FRESH_S):
    """Knowledge table at time t: the last snapshot, and the ids that changed in it while it is still fresh."""
    snaps = [e for e in timeline if e['kind'] == 'kb_snapshot' and e['t'] <= t]
    if not snaps: return dict(claims=[], changed=[], t=None)
    current = snaps[-1]; before = {c['id']: c for c in (snaps[-2]['claims'] if len(snaps) > 1 else [])}
    changed = [c['id'] for c in current['claims'] if before.get(c['id']) != c] if t - current['t'] <= fresh else []
    return dict(claims=current['claims'], changed=changed, t=current['t'])


def badge_intervals(timeline):
    """[(t0, t1, badge)]: a hold lasts until the next resume (or the end); a masked attack shows for MASK_S."""
    out = []
    for i, e in enumerate(timeline):
        if e['kind'] == 'hold':
            end = next((r['t'] for r in timeline[i + 1:] if r['kind'] == 'resume'), float('inf'))
            latency = e.get('hold_latency_ms') or 0.
            if str(e.get('reason', '')).startswith('guard_'):
                text = '⏸ 파괴 전 정지 (%.2f ms · 채굴 경과 %d틱)' % (latency, e.get('mining_ticks_at_hold') or 0)
            else: text = '⏸ 즉시 정지 (%.2f ms)' % latency
            out.append((e['t'], end, dict(kind='hold', text=text, reason=e.get('reason'))))
        elif e['kind'] == 'monitor_event' and (e.get('event') or {}).get('kind') == 'guard_violation':
            out.append((e['t'], e['t'] + MASK_S, dict(kind='mask', text='⛔ 타격 차단', reason='guard_violation')))
    return out


def badge_at(timeline, t):
    active = [b for t0, t1, b in badge_intervals(timeline) if t0 <= t <= t1]
    return next((b for b in active if b['kind'] == 'hold'), active[0] if active else None)


def rm_at(timeline, t):
    """Plan steps with done/active/held/pending status at time t (an inserted sub-goal keeps the pending rest)."""
    plans = [(e['t'], e.get('plan') if e['kind'] == 'session_started' else e.get('steps'), e.get('mode', 'replan'))
             for e in timeline if e['kind'] in ('session_started', 'plan') and e['t'] <= t]
    if not plans: return []
    def statuses(since, until):
        state = {}
        for e in timeline:
            if not since <= e['t'] <= until: continue
            if e['kind'] == 'hold': state[e.get('step_id')] = 'held'  # step_finished comes only after the slow path
            elif e['kind'] in ('step_started', 'step_finished'):
                state[e['id']] = 'active' if e['kind'] == 'step_started' else (
                    'done' if e.get('status') == 'completed' else 'held' if e.get('status') == 'anomaly_hold' else 'failed')
        return state
    steps = []
    for i, (since, plan, mode) in enumerate(plans):
        until = plans[i + 1][0] if i + 1 < len(plans) else t
        state = statuses(since, until)
        current = [dict(id=s['id'], target=s['target'], count=s['count'], status=state.get(s['id'], 'pending')) for s in plan]
        if mode == 'insert': current += [s for s in steps if s['status'] == 'pending']
        steps = current
    return steps


def describe(e):
    """(text, category) for the 'what just happened' list and the timeline marks; None for bookkeeping events."""
    kind = e['kind']; ev = e.get('event') or {}
    if kind == 'anomaly':
        block = ev.get('block') or (ev.get('evidence') or {}).get('block')
        return '이상 감지: %s%s' % (ev.get('kind'), ' (%s)' % block if block else ''), 'anomaly'
    if kind == 'hold': return '정지(hold): %s' % e.get('reason'), 'anomaly'
    if kind == 'kb_query': return '지식 조회: %s %s' % (e.get('subject'), e.get('relation')), 'knowledge'
    if kind == 'slow_path':
        return '느린 판단: %s → %s' % (ROUTES.get(e.get('route'), e.get('route')), e.get('resolution')), 'knowledge'
    if kind == 'llm_request':
        what = 'LLM 조건 가설 요청' if e.get('purpose') == 'condition_hypothesis' else 'LLM 웹 검색 요청'
        return '%s (%s)' % (what, e.get('model')), 'knowledge'
    if kind == 'kb_change':
        if e.get('evidence') == 'game_observation_support': return '게임 관측 지지 근거 +%d' % len(e.get('claim_ids') or []), 'knowledge'
        return '지식 추가 %d개 (거부 %d)' % (len(e.get('claim_ids') or []), len(e.get('rejected') or [])), 'knowledge'
    if kind == 'plan':
        label = '하위 목표 삽입' if e.get('mode') == 'insert' else '재계획'
        return '%s: %s' % (label, ' → '.join(s['id'] for s in e.get('steps') or [])), 'replan'
    if kind == 'resume': return '재개 (감지→재개 %.1f초)' % (e.get('detect_to_resume_s') or 0.), 'replan'
    if kind == 'step_started': return '단계 시작: %s (%s)' % (e.get('id'), e.get('skill')), 'info'
    if kind == 'step_finished':
        if e.get('status') == 'completed': return '단계 완료: %s' % e.get('id'), 'success'
        return '단계 중단: %s (%s)' % (e.get('id'), e.get('status')), 'anomaly'
    if kind == 'monitor_event':
        if ev.get('kind') == 'guard_violation': return '⛔ 타격 차단: %s (도구 부족)' % ev.get('block', ''), 'risk'
        return '감시 기록: %s' % ev.get('kind'), 'info'
    if kind == 'knowledge_conflict':
        return '지식 충돌: %s drops %s ↔ 관측 %s → %s' % (e.get('subject'), e.get('expected'), e.get('observed'), e.get('outcome')), 'knowledge'
    if kind == 'learning_request': return '학습 요청 기록: %s (실행 안 함)' % e.get('rule_skill'), 'knowledge'
    if kind == 'gui_closed': return 'GUI 닫기 (세션)', 'info'
    return None


def events_at(timeline, t, n=8):
    out = []
    for e in timeline:
        if e['t'] > t: break
        d = describe(e)
        if d: out.append(dict(t=e['t'], text=d[0], category=d[1]))
    return out[-n:]


def _place(src, dst):
    if dst.exists(): return
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)


def build_replay(run_dir, out_dir):
    """Write out_dir/replay.html (data inline, frames next to it) and replay-data.json; returns the HTML path."""
    run_dir, out_dir = Path(run_dir), Path(out_dir); (out_dir / 'frames').mkdir(parents=True, exist_ok=True)
    timeline = read_jsonl(run_dir / 'timeline.jsonl'); samples = {r['step']: r for r in read_jsonl(run_dir / 'telemetry.jsonl')}
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8')) if (run_dir / 'summary.json').exists() else {}
    frames = []
    for path in sorted((run_dir / 'frames').glob('*.jpg')):
        step = int(path.stem); sample = samples.get(step)
        if sample is None: continue
        _place(path, out_dir / 'frames' / path.name)
        frames.append(dict(step=step, t=sample['t'], file='frames/' + path.name, controller=sample.get('controller'),
                           inventory={k: v for k, v in (sample.get('inventory') or {}).items() if v},
                           selected=sample.get('server_selected_item'), badge=badge_at(timeline, sample['t'])))
    marks = sorted({e['t'] for e in timeline if e['kind'] in ('plan', 'session_started', 'step_started', 'step_finished')})
    data = dict(
        meta=dict(run=run_dir.name, status=summary.get('status'), final_inventory=summary.get('final_inventory'),
                  api_calls=(summary.get('oversight') or {}).get('api_calls'), label='준비된 시연 월드',
                  t0=frames[0]['t'] if frames else 0., t1=max([f['t'] for f in frames] + [e['t'] for e in timeline] or [0.])),
        frames=frames,
        kb=[dict(t=e['t'], claims=e['claims']) for e in timeline if e['kind'] == 'kb_snapshot'],
        badges=[dict(t0=t0, t1=None if t1 == float('inf') else t1, **b) for t0, t1, b in badge_intervals(timeline)],
        events=events_at(timeline, float('inf'), n=10 ** 9), rm=[dict(t=t, steps=rm_at(timeline, t)) for t in marks])
    (out_dir / 'replay-data.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    inline = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    page = PAGE.replace('__TITLE__', html.escape(run_dir.name)).replace('__DATA__', inline)
    out = out_dir / 'replay.html'; out.write_text(page, encoding='utf-8')
    return out


PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>시연 재생 · __TITLE__</title>
<style>
:root{--bg:#f6f5f2;--panel:#ffffff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--accent:#2f6fde;--hold:#c2410c;--mask:#b91c1c;
--ok:#15803d;--know:#7c3aed;--replan:#0e7490;--risk:#b91c1c;--fresh:#fff4c2}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--panel:#1d1d21;--ink:#ececf1;--muted:#a0a0aa;
--line:#2e2e35;--accent:#7aa7ff;--hold:#fb923c;--mask:#f87171;--ok:#4ade80;--know:#c4b5fd;--replan:#67e8f9;--risk:#f87171;--fresh:#3d3412}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
header{display:flex;gap:12px;align-items:baseline;padding:12px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0}.tag{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}
.tag.world{border-color:var(--accent);color:var(--accent)}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.view{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:16/9}
.view img{width:100%;height:100%;object-fit:contain;display:block}
.ctl{position:absolute;left:10px;top:10px;background:rgba(0,0,0,.65);color:#fff;padding:4px 10px;border-radius:6px;font-weight:600}
.badge{position:absolute;left:50%;top:10px;transform:translateX(-50%);padding:6px 14px;border-radius:8px;color:#fff;font-weight:700;display:none}
.badge.hold{background:var(--hold)}.badge.mask{background:var(--mask)}
.inv{position:absolute;left:0;right:0;bottom:0;display:flex;flex-wrap:wrap;gap:6px;padding:8px;background:linear-gradient(transparent,rgba(0,0,0,.75))}
.inv span{color:#fff;font-size:12px;background:rgba(255,255,255,.15);padding:2px 7px;border-radius:5px}
.inv span.sel{outline:2px solid #fff}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
section h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600}
ol.rm{margin:0;padding-left:20px}ol.rm li{margin:2px 0}.st{font-size:11px;padding:1px 6px;border-radius:4px;margin-left:6px;border:1px solid var(--line)}
.st.done{color:var(--ok);border-color:var(--ok)}.st.active{color:var(--accent);border-color:var(--accent)}.st.held,.st.failed{color:var(--hold);border-color:var(--hold)}
ul.ev{list-style:none;margin:0;padding:0}ul.ev li{padding:3px 0;border-bottom:1px dashed var(--line);display:flex;gap:8px}
ul.ev time{color:var(--muted);font-variant-numeric:tabular-nums;min-width:52px}
.c-anomaly{color:var(--hold)}.c-knowledge{color:var(--know)}.c-replan{color:var(--replan)}.c-success{color:var(--ok)}.c-risk{color:var(--risk)}
table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}tr.fresh td{background:var(--fresh)}td.src a{color:var(--accent);word-break:break-all}
.s-supported{color:var(--ok)}.s-disputed,.s-retracted{color:var(--hold)}.s-retracted{text-decoration:line-through}
footer{padding:0 16px 16px}.bar{position:relative;height:28px;background:var(--panel);border:1px solid var(--line);border-radius:8px;cursor:pointer}
.bar i{position:absolute;top:4px;width:3px;height:20px;border-radius:2px}.bar b{position:absolute;top:0;bottom:0;width:2px;background:var(--ink)}
.controls{display:flex;gap:10px;align-items:center;margin-top:8px}button,select{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.legend{display:flex;gap:12px;font-size:12px;color:var(--muted);margin-left:auto}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style></head><body>
<header><h1>철 캐줘 — 시연 재생</h1><span class="tag world">준비된 시연 월드</span><span class="tag" id="run"></span><span class="tag" id="status"></span></header>
<main>
 <div>
  <div class="view"><img id="frame" alt="에이전트 시점"><div class="ctl" id="ctl"></div><div class="badge" id="badge"></div><div class="inv" id="inv"></div></div>
 </div>
 <div>
  <section><h2>RM 진행</h2><ol class="rm" id="rm"></ol></section>
  <section><h2>방금 일어난 일</h2><ul class="ev" id="ev"></ul></section>
  <section><h2>지식 (SQLite)</h2><table><thead><tr><th>주장</th><th>조건</th><th>상태</th><th>근거</th></tr></thead><tbody id="kb"></tbody></table></section>
 </div>
</main>
<footer><div class="bar" id="bar"><b id="cursor"></b></div>
<div class="controls"><button id="play">▶ 재생</button><select id="speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select>
<span id="clock"></span><div class="legend"><span><i style="background:var(--risk)"></i>위험 감지</span><span><i style="background:var(--hold)"></i>이상</span>
<span><i style="background:var(--know)"></i>지식</span><span><i style="background:var(--replan)"></i>재계획</span><span><i style="background:var(--ok)"></i>성공</span></div></div></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),M=D.meta,T0=M.t0,T1=Math.max(M.t1,T0+1);
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const CTL={policy:'정책 (STEVE-1)',rule:'규칙 스킬',gui_close:'세션: GUI 닫기',prepare:'준비',hold:'정지'},ST={proposed:'제안',supported:'지지',disputed:'논쟁',retracted:'철회'};
const RMS={done:'완료',active:'진행',held:'정지',failed:'실패',pending:'대기'},COL={risk:'--risk',anomaly:'--hold',knowledge:'--know',replan:'--replan',success:'--ok'};
$('run').textContent=M.run;$('status').textContent='결과: '+M.status+(M.api_calls!=null?' · API '+M.api_calls+'회':'');
const last=(a,t,k='t')=>{let lo=0,hi=a.length-1,r=-1;while(lo<=hi){const m=(lo+hi)>>1;if(a[m][k]<=t){r=m;lo=m+1}else hi=m-1}return r};
for(const e of D.events){if(!COL[e.category])continue;const i=document.createElement('i');i.style.left=((e.t-T0)/(T1-T0)*100)+'%';i.style.background='var('+COL[e.category]+')';i.title=e.text;$('bar').appendChild(i)}
const cond=c=>Object.keys(c).length?Object.entries(c).map(([k,v])=>k+' '+(typeof v==='object'?Object.entries(v).map(([o,x])=>(o==='gte'?'≥':'≤')+x).join(' '):v)).join(', '):'항상';
const obj=o=>typeof o==='object'&&o!==null?Object.entries(o).map(([k,v])=>k+'='+JSON.stringify(v)).join(' '):JSON.stringify(o);
let now=T0,playing=false,lastWall=null;
function render(){
 const fi=last(D.frames,now),f=D.frames[Math.max(fi,0)];
 if(f){$('frame').src=f.file;$('ctl').textContent=CTL[f.controller]||f.controller;
  $('inv').innerHTML=Object.entries(f.inventory).map(([k,v])=>'<span'+(k===f.selected?' class="sel"':'')+'>'+esc(k)+' ×'+v+'</span>').join('')}
 const b=D.badges.filter(x=>x.t0<=now&&(x.t1===null||now<=x.t1)),bh=b.find(x=>x.kind==='hold')||b[0];
 $('badge').style.display=bh?'block':'none';if(bh){$('badge').className='badge '+bh.kind;$('badge').textContent=bh.text}
 const ri=last(D.rm,now);$('rm').innerHTML=ri<0?'':D.rm[ri].steps.map(s=>'<li>'+esc(s.id)+' <small>('+esc(s.target)+' ×'+s.count+')</small><span class="st '+s.status+'">'+RMS[s.status]+'</span></li>').join('');
 const ei=last(D.events,now);$('ev').innerHTML=D.events.slice(Math.max(0,ei-7),ei+1).map(e=>'<li class="c-'+e.category+'"><time>'+(e.t-T0).toFixed(1)+'s</time><span>'+esc(e.text)+'</span></li>').join('');
 const ki=last(D.kb,now);let rows='';
 if(ki>=0){const cur=D.kb[ki],prev=ki>0?Object.fromEntries(D.kb[ki-1].claims.map(c=>[c.id,JSON.stringify(c)])):{},fresh=now-cur.t<=3;
  for(const c of cur.claims){const ch=fresh&&prev[c.id]!==JSON.stringify(c);const src=(c.sources||[]).slice(0,2).map(u=>'<a href="'+esc(u)+'">'+esc(u.replace(/^https?:\/\//,''))+'</a>').join('<br>');
   const g=c.observed||{};rows+='<tr'+(ch?' class="fresh"':'')+'><td>'+esc(c.subject+' '+c.relation+' '+obj(c.object))+'</td><td>'+esc(cond(c.conditions))+'</td><td class="s-'+c.status+'">'+(ST[c.status]||c.status)+'</td><td class="src">'+src+(g.support||g.refute?'<div>게임 관측 +'+(g.support||0)+' / −'+(g.refute||0)+'</div>':'')+'</td></tr>'}}
 $('kb').innerHTML=rows||'<tr><td colspan="4" style="color:var(--muted)">비어 있음</td></tr>';
 $('cursor').style.left=((now-T0)/(T1-T0)*100)+'%';$('clock').textContent=(now-T0).toFixed(1)+' / '+(T1-T0).toFixed(1)+' 초';
}
function tick(w){if(!playing)return;if(lastWall!==null)now=Math.min(T1,now+(w-lastWall)/1000*+$('speed').value);lastWall=w;render();if(now>=T1){playing=false;$('play').textContent='▶ 재생'}else requestAnimationFrame(tick)}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'⏸ 멈춤':'▶ 재생';lastWall=null;if(playing){if(now>=T1)now=T0;requestAnimationFrame(tick)}};
$('bar').onclick=e=>{const r=$('bar').getBoundingClientRect();now=T0+(e.clientX-r.left)/r.width*(T1-T0);render()};
document.addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();$('play').click()}if(e.key==='ArrowRight'){now=Math.min(T1,now+2);render()}if(e.key==='ArrowLeft'){now=Math.max(T0,now-2);render()}});
render();
</script></body></html>
'''
