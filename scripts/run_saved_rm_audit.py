"""Audit one preserved trajectory in WSL; no game, API, training, or STOP rearm."""
import argparse,json,subprocess,time
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--label',default='audit');a=p.parse_args()
    run=a.run.resolve();out=a.output.resolve()
    if not run.is_relative_to(ROOT/'results') or not out.is_relative_to(ROOT/'results'):
        raise ValueError('project-local evidence required')
    if not a.label.replace('-','').isalnum():raise ValueError('simple audit label required')
    out.mkdir(parents=True,exist_ok=True);log=out/(a.label+'.log');receipt=out/(a.label+'.json')
    if log.exists() or receipt.exists():raise FileExistsError('existing audit attempt preserved')
    probe="""import os,sys,time,faulthandler,runpy
os.dup2(1,2)
print('PYTHON_STARTED',time.time(),flush=True)
faulthandler.dump_traceback_later(60,repeat=True)
start=time.monotonic();import torch
print('TORCH_IMPORTED',round(time.monotonic()-start,3),'default_threads',torch.get_num_threads(),flush=True)
torch.set_num_threads(1)
original=torch.load
def timed_load(path,*a,**kw):
 start=time.monotonic();print('LOAD_STARTED',str(path),flush=True);value=original(path,*a,**kw);print('LOAD_DONE',round(time.monotonic()-start,3),flush=True);return value
torch.load=timed_load
sys.path.insert(0,'scripts')
sys.argv=['scripts/audit_fresh_target_handoff.py',sys.argv[1],sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
faulthandler.cancel_dump_traceback_later()
"""
    cmd=['wsl.exe','-d','Ubuntu-24.04','--cd','/mnt/d/DATA/Desktop/MinecraftRL','--','env',
         'PYTHONPATH=src','LD_LIBRARY_PATH=/usr/lib/wsl/lib','/usr/bin/timeout','--kill-after=15','600',
         '/mnt/d/DATA/Desktop/MinecraftRL/.runtime/steve1/bin/python','-u','-c',probe,
         run.relative_to(ROOT).as_posix(),'results/onset-coverage-training-20261009/checkpoint.pt']
    started=time.monotonic();result=dict(completed=False,exit_code=None,error=None,run=run.relative_to(ROOT).as_posix(),
        api_calls=0,game_actions=0,scope='Preserved trajectory; one torch CPU thread; no gameplay')
    try:
        with log.open('x',encoding='utf-8') as stream:
            proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,timeout=750)
        result.update(completed=proc.returncode==0,exit_code=proc.returncode)
    except Exception as exc:result['error']=repr(exc)
    finally:
        result.update(elapsed_s=time.monotonic()-started,log_sha256=digest(log))
        write_json(receipt,result)
    print(json.dumps(result));return 0 if result['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
