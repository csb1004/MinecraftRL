"""Small read-only progress view; never dump large raw observation lines."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root/'results/goal-recovery-comparison-20261009'
state = json.loads((base/'experiment.json').read_text(encoding='utf-8-sig'))
print(json.dumps(dict(completed=state['completed'],active=state['active'],errors=state['errors'],
    runs=[dict(label=r['label'],audited=r['audited'],status=r['report']['status'],
          steps=r['report']['steps'],wood=r['report'].get('server_target_reached'),
          intervention=r['report'].get('mode_steps')) for r in state['runs']])))
if state['active']:
    retry = any(t['label']==state['active'] for t in state.get('technical_attempts',[]))
    log = base/(state['active']+('-retry-startup-1' if retry else '')+'.log')
    if log.exists():
        with log.open(encoding='utf-8') as src: first = src.readline().strip()
        if first:
            run = Path(first)/'run'; events = run/'events.jsonl'
            print('run='+str(run))
            if not run.exists():
                print('phase=supervisor_startup'); raise SystemExit(0)
            print('artifacts='+','.join(p.name for p in run.iterdir() if p.is_file()))
            if events.exists():
                with events.open('rb') as src:
                    size = src.seek(0,2); src.seek(max(0,size-2**21)); chunk = src.read()
                lines = chunk.split(b'\n')[1:-1] if size>2**21 else chunk.split(b'\n')[:-1]
                rows = []
                for line in lines:
                    try: e = json.loads(line)
                    except (ValueError,UnicodeDecodeError): continue
                    if e.get('event') == 'transition':
                        s = e['info']['workshop']
                        rows.append(dict(step=e['step'],controller=e['controller'],
                            xyz=[s['server_'+k] for k in 'xyz'],yaw=s['yaw'],
                            inventory={k:v for k,v in s['server_inventory'].items() if v}))
                print(json.dumps(rows[-1:] or {'tail_complete_transition':False}))
