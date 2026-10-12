"""Compact read-only status, including bounded tails of live logs."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=Path(__file__).resolve().parents[1]/'results/recovery-curriculum-20261007')
base=p.parse_args().batch
state=json.loads((base/'experiment.json').read_text());active=state['active']
result=dict(completed=state['completed'],active=active,errors=state['errors'],
            runs=[dict(label=r['label'],status=r['report']['status'],steps=r['report']['steps'],audited=r['audited']) for r in state['runs']],
            updates=sum(t['report']['updates'] for t in state.get('training',[])))
if active:
    if active.startswith('train-') and (base/active/'updates.jsonl').exists():
        file=base/active/'updates.jsonl'
        if file.exists():
            with file.open('rb') as f:f.seek(max(0,file.stat().st_size-20000));lines=f.read().splitlines()
            for line in reversed(lines):
                try:result['optimizer_latest']=json.loads(line);break
                except ValueError:pass
    else:
        log=base/(active+'.log')
        if log.exists() and log.stat().st_size:
            with log.open() as f:run=Path(f.readline().strip())/'run'
            result['run']=str(run);file=run/'events.jsonl'
            if file.exists() and file.stat().st_size:
                with file.open('rb') as f:f.seek(max(0,file.stat().st_size-1500000));lines=f.read().splitlines()
                for line in reversed(lines):
                    try:e=json.loads(line)
                    except ValueError:continue
                    if e.get('event')!='transition':continue
                    s=e['info']['workshop'];part=e['info']['rl_reward']
                    result['latest']=dict(server_tick=s['server_tick'],stage=part['stage'],
                        recovery_active=part['recovery']['active'],logs=sum(s['server_inventory'].get(k+'_log',0) for k in ('oak','birch','spruce','jungle','acacia','dark_oak')),
                        position=[s['server_'+k] for k in 'xyz'])
                    break
print(json.dumps(result))
