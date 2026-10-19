"""Six predeclared development games on one frozen source, with per-run audits."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/local-navigation-reliability-20261019'
LINUX='/mnt/c/MinecraftRL'


def sources():
    return {p.relative_to(ROOT/'src').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'src').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def run(argv,log):
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return result.returncode


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.results_root.resolve()
    if not OUT.is_relative_to(ROOT/'results'):raise ValueError('project-local results required')
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    planned=[dict(label=f'verify-{kind}-{i:02d}',seed=seed,kind=kind)
        for kind in ('blind','known') for i,seed in enumerate(protocol['verification']['seeds'],1)]
    freeze=dict(source_sha256=sources(),runs=planned,scope='Six development repetitions; not final-test seeds')
    with (OUT/'verification-freeze.json').open('x',encoding='utf-8') as f:json.dump(freeze,f,indent=2)
    outcomes=[]
    for item in planned:
        if (OUT/'HALT').exists():
            print('Batch halted before next game; completed outcomes preserved.',flush=True);return
        if sources()!=freeze['source_sha256']:raise ValueError('source changed during frozen verification')
        label=item['label'];attempt=OUT/label
        argv=[sys.executable,str(ROOT/'scripts/run_local_navigation_comparison.py'),label,
            '--strategy','local_passability','--sampling-seed',str(item['seed']),'--output-root',str(OUT)]
        if item['kind']=='known':argv+=['--world',str(ROOT/'results/local-navigation-detour-20261019/worlds/known-corner.zip')]
        print('Starting '+label,flush=True)
        exit_code=run(argv,OUT/(label+'-launcher.log'))
        if exit_code or not (attempt/'run/summary.json').exists():raise RuntimeError(label+' launcher failed; inspect preserved output')
        actual=json.loads((attempt/'protocol.json').read_text())['source_sha256']
        if actual!=freeze['source_sha256']:raise ValueError('execution source differs from freeze')
        runtime=json.loads((attempt/'run/world-runtime.json').read_text())['path']
        if not runtime.startswith(LINUX+'/data/worlds/runtime/'):raise ValueError('unexpected runtime root')
        runtime=ROOT/runtime[len(LINUX)+1:]
        if run([sys.executable,str(ROOT/'scripts/preserve_local_navigation_world.py'),str(attempt),str(runtime)],attempt/'preservation.log'):
            raise RuntimeError('world preservation failed')
        if run([sys.executable,str(ROOT/'scripts/report_local_reliability.py'),str(attempt)],attempt/'reliability.log'):
            raise RuntimeError('outcome report failed')
        audit=['wsl.exe','-d','Ubuntu-24.04','--cd',LINUX,'--','env','PYTHONPATH=src','PYTHONDONTWRITEBYTECODE=1',
            '.runtime/bin/micromamba','run','-p','.runtime/steve1','python','scripts/audit_local_navigation_run.py',
            attempt.relative_to(ROOT).as_posix()]
        if run(audit,attempt/'audit.log'):raise RuntimeError(label+' independent audit failed')
        result=json.loads((attempt/'reliability.json').read_text());outcomes.append(result)
        (OUT/'verification-results.json').write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('label','normal_episode_success','elapsed_control_s','server_log_gain')}),flush=True)
        if result['status'] in ('cancelled','error'):raise RuntimeError('user stop/runtime error: no next game')
    print('Verification finished: '+str(sum(r['normal_episode_success'] for r in outcomes))+'/'+str(len(outcomes)),flush=True)


if __name__=='__main__':main()
