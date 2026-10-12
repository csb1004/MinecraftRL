"""Windows entry point for one bounded, persistent STEVE-1 dialogue session."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from .dialogue_console import SessionDialogue
from .memory import atomic_json
from .semantic import OpenAISemantic


HELP = ('명령: 흙 2개 모아줘 / 원목 2개 모아줘 / 목표를 흙 3개로 바꿔줘 / '
        '잠시 멈춰줘 / 다시 계속해줘 / 작업 취소해줘\n'
        '/status 상태 · /help 도움말 · /exit 또는 /stop 저장 후 종료\n'
        '수량은 추가 획득량입니다. 조합·이동·제작은 현재 미지원입니다. '
        '기본 모드는 제한된 한국어 명령이며 API를 사용하지 않습니다.')


def describe_task(task):
    if not task: return '대기: 아직 목표가 없습니다. 예: 흙 2개 모아줘'
    status={'running':'수행 중','paused':'일시정지','completed':'완료','cancelled':'취소'}.get(task['status'],task['status'])
    goal={'wood':'원목','dirt':'흙'}.get(task['goal'],task['goal'])
    text=f"{status}: {goal} {task['progress']}/{task['target_count']} · 완료한 단계 {len(task.get('completed_stages',[]))}"
    if task.get('failure',{}).get('reason')=='no_progress': text+=' · 정체: 목표 획득 증가가 없어 멈췄습니다.'
    if task['status']=='paused': text+=' 재개하려면: 다시 계속해줘'
    if task['status'] in ('completed','cancelled'): text+=' 새 수집 목표를 요청할 수 있습니다.'
    return text


def worker_command(linuxroot,distribution,slot,seed,seconds,relative_output):
    return ['wsl.exe','-d',distribution,'--cd',linuxroot,'--','env','GALLIUM_DRIVER=d3d12',
            'LD_LIBRARY_PATH=/usr/lib/wsl/lib','MAMBA_ROOT_PREFIX='+linuxroot+'/.runtime/mamba',
            'XDG_CACHE_HOME='+linuxroot+'/.cache','HF_HOME='+linuxroot+'/.cache/huggingface',
            linuxroot+'/.runtime/bin/micromamba','run','-p',linuxroot+'/.runtime/steve1',
            'python','-m','minecraftrl.cli','session','--root',linuxroot,'--policy','steve1',
            '--seed',str(seed),'--limit-s',str(seconds),'--world-slot',slot,
            '--output',linuxroot+'/'+relative_output,'--stop-file',linuxroot+'/results/STOP']


def read_input(inbox):
    while True:
        line=sys.stdin.readline()
        inbox.put(line.rstrip('\r\n') if line else '/exit')
        if not line: return


def finish_worker(proc,stop,*,natural_end):
    if not natural_end: stop.touch()
    try:
        if proc is not None: proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        stop.touch()
        return False
    stop.touch()
    return True


def interact(session,proc,stop,inbox):
    previous=None
    print(HELP,flush=True)
    while proc.poll() is None and not stop.exists():
        try:
            task=session.sync()
        except ValueError as exc:
            if str(exc)=='session_stopped_or_ended': break
            raise
        status=describe_task(task)
        if status!=previous: print(status,flush=True); previous=status
        try: text=inbox.get(timeout=.2)
        except queue.Empty: continue
        if text in ('/exit','/stop'): break
        if text=='/help': print(HELP,flush=True); continue
        if text=='/status': print(status,flush=True); continue
        if not text.strip(): continue
        try:
            result=session.turn(text)
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except (ValueError,RuntimeError) as exc: print(str(exc),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-slot',default='companion')
    parser.add_argument('--user-id',default='local-operator')
    parser.add_argument('--distribution',default='Ubuntu-24.04')
    parser.add_argument('--seed',type=int,default=104729)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--allow-paid-api',action='store_true')
    parser.add_argument('--max-api-calls',type=int,default=5)
    args=parser.parse_args()
    if os.name!='nt': parser.error('Start this entry point from Windows PowerShell.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.world_slot): parser.error('invalid world slot')
    if not 1<=args.seconds<=180: parser.error('seconds must be 1..180')
    if not 1<=args.max_api_calls<=20: parser.error('max-api-calls must be 1..20')
    root=Path(__file__).resolve().parents[2]; results=root/'results'; results.mkdir(exist_ok=True)
    # Validate provider before launching/re-arming. No request occurs in construction.
    provider=OpenAISemantic(os.getenv('OPENAI_MODEL'),os.getenv('OPENAI_API_KEY'),
        allow_paid=True,max_calls=args.max_api_calls) if args.allow_paid_api else None
    linuxroot=subprocess.check_output(['wsl.exe','-d',args.distribution,'--','wslpath','-a',root.as_posix()],text=True,stdin=subprocess.DEVNULL).strip()
    lock=results/'companion-launch.lock'
    try: lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: parser.error('Another launcher or an interrupted launch owns results/companion-launch.lock. Inspect its process before removing that lock.')
    os.close(lock_fd)
    stop=results/'STOP'; proc=None; folder=None; session=None
    try:
        if not stop.exists(): raise RuntimeError('STOP is not latched. Finish the existing session before starting another.')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder=results/('companion-'+stamp); folder.mkdir()
        lock.write_text(json.dumps({'pid':os.getpid(),'session':str(folder)}),encoding='utf-8')
        identity=hashlib.sha256(args.user_id.encode('utf-8')).hexdigest()[:32]
        memory=root/'data/dialogue'/(identity+'.json'); memory.parent.mkdir(parents=True,exist_ok=True)
        run=folder/'run'
        session=SessionDialogue(run,provider,persona=(root/'configs/dialogue-persona.txt').read_text(encoding='utf-8-sig'),
                                memory_path=memory,stop_file=stop)
        atomic_json(folder/'launcher.json',dict(world_slot=args.world_slot,user_identity_hash=identity,
                    paid_api_enabled=args.allow_paid_api,max_api_calls=args.max_api_calls if provider else 0))
        print(f'월드: {args.world_slot}\n기록: {folder}\n게임·정책 초기화 중입니다. 준비 후 입력하세요. Ctrl+C로 중단할 수 있습니다.',flush=True)
        with (folder/'supervisor.log').open('w',encoding='utf-8') as log:
            stop.rename(folder/'STOP-before.txt')
            proc=subprocess.Popen(worker_command(linuxroot,args.distribution,args.world_slot,args.seed,args.seconds,
                                  run.relative_to(root).as_posix()),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            deadline=time.monotonic()+600
            while proc.poll() is None and not stop.exists():
                try: session.sync(); break
                except ValueError as exc:
                    if str(exc)!='session_not_ready': raise
                if time.monotonic()>deadline: raise RuntimeError('initialization timeout')
                time.sleep(.2)
            if proc.poll() is None and not stop.exists():
                inbox=queue.Queue()
                threading.Thread(target=read_input,args=(inbox,),daemon=True).start()
                interact(session,proc,stop,inbox)
    except KeyboardInterrupt:
        print('중단 요청을 전달합니다.',flush=True)
    finally:
        # Never re-arm on exit. The WSL supervisor owns process cleanup and world saving.
        if folder is not None:
            # A naturally finished worker must finish its receipt before STOP is
            # re-latched, otherwise the supervisor can relabel success as cancellation.
            natural_end=(folder/'run/summary.json').exists() or (session is not None and session.closed)
            print('입력 해제·월드 저장·프로세스 종료 대기 중…',flush=True)
            if not finish_worker(proc,stop,natural_end=natural_end):
                print('종료 확인 시간 초과. STOP과 잠금 파일을 유지합니다. supervisor.log를 확인하세요.',flush=True)
                return 2
        lock.unlink()
    if folder is None: return 2
    summary=folder/'run/summary.json'
    if not summary.exists(): print('종료 결과가 없습니다. '+str(folder/'supervisor.log'),flush=True); return 2
    report=json.loads(summary.read_text(encoding='utf-8'))
    clean=(report.get('release')=='neutral_command_sent' and report.get('world_save',{}).get('status')=='saved'
           and report.get('process_cleanup',{}).get('remaining_live_pids')==[] and 'error' not in report)
    receipt=dict(status=report['status'],task=report.get('task'),clean_shutdown=clean,session=str(folder))
    atomic_json(folder/'launcher-result.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    print('같은 명령과 월드 이름으로 다시 실행하면 저장된 과업을 불러옵니다.',flush=True)
    return 0 if clean else 2


if __name__=='__main__': raise SystemExit(main())
