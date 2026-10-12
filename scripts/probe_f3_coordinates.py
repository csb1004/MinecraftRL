"""One isolated F3 UI diagnostic. No neural policy, learning or paid provider."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from minecraftrl.runtime import write_json
from minecraftrl.rl_control import claim_automatic_stop,publish_completion


def worker(root,out):
    from PIL import Image
    from minerl.env.malmo import InstanceManager
    from minerl.utils import process_watcher
    from minerl.herobraine.hero.handlers.agent.actions.keyboard import KeybasedCommandAction
    from minecraftrl.workshop_monitor import workshop_survival
    from minecraftrl.minerl_runtime import prepare_jar_cache,launch_scoped_watcher,configure_player_identity
    from minecraftrl.world_save_patch import build_save_jar
    from minecraftrl.policies import MineRLEnvironment
    stop=root/'results/STOP'
    process_watcher.launch=partial(launch_scoped_watcher,out)
    mc=Path(InstanceManager.MINECRAFT_DIR)
    prepare_jar_cache(mc,out,source_override=build_save_jar(root,mc,craft_monitor=True,workshop_monitor=True))
    configure_player_identity(mc,'f3-coordinate-diagnostic')
    base=workshop_survival()
    class DebugSurvival(type(base)):
        def create_actionables(self):
            # Upstream actionToKey maps the suffix of hotbar.* to a literal GLFW key.
            # This private diagnostic alias is NOT part of the learned action space.
            return super().create_actionables()+[KeybasedCommandAction('hotbar.f3','key.keyboard.f3')]
    raw=DebugSurvival(fov_range=[70,70],frameskip=1,gamma_range=[2,2],guiscale_range=[2,2],
        resolution=[1280,720],cursor_size_range=[16.,16.],
        load_filename=str(root/'results/diverse-teacher-worlds-20261008/development-corner.zip')).make()
    env=MineRLEnvironment(raw);rows=[];start=None
    try:
        raw.seed(104761);obs=env.reset();start=time.monotonic()
        Image.fromarray(obs['pov']).save(out/'initial.png')
        actions=[('settle',0),('f3_press',1),('f3_release',0),('f3_visible',0),
                 ('f3_close_press',1),('f3_close_release',0),('f3_closed',0)]
        for name,key in actions:
            if stop.exists() or time.monotonic()-start>30:raise InterruptedError('F3 probe stopped')
            action=raw.action_space.noop();action['hotbar.f3']=key
            obs,_,done,info=env.step(action)
            Image.fromarray(obs['pov']).save(out/(name+'.png'))
            rows.append(dict(stage=name,f3_key=key,server=info['workshop'],done=bool(done)))
            write_json(out/'observations.json',rows)
            if done:raise RuntimeError('environment ended during F3 diagnostic')
        write_json(out/'capture.json',dict(completed=True,steps=len(rows),elapsed_control_s=time.monotonic()-start,
            scope='UI-only diagnostic alias hotbar.f3 invokes native F3 release; no learned policy',
            policy_actions=0,api_calls=0,optimizer_updates=0))
    finally:
        release=env.release();write_json(out/'release.json',dict(result=release,evidence=getattr(env,'release_evidence',None)))
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project result directory required')
    if a.worker:worker(root,out);return
    from minecraftrl.processes import ProcessOwner
    out.mkdir(parents=True,exist_ok=False);stop=root/'results/STOP';claim_automatic_stop(stop,out/'STOP-before.txt')
    owner=ProcessOwner();proc=None;result=dict(completed=False,error=None,api_calls=0,new_policy_games=0)
    try:
        with (out/'console.log').open('w') as stream:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(out),'--worker'],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            deadline=time.monotonic()+360
            while proc.poll() is None:
                owner.capture()
                if stop.exists() or time.monotonic()>deadline:
                    stop.touch(exist_ok=True);raise TimeoutError('F3 probe closed/expired')
                time.sleep(.1)
        if proc.returncode:raise RuntimeError('F3 worker failed; inspect preserved console')
        result['completed']=True
    except Exception as exc:result['error']=repr(exc)
    finally:
        if proc:result['cleanup']=owner.close(proc)
        publish_completion(stop,'RL collection ended; F3 coordinate diagnostic finished.')
        write_json(out/'work.json',result)
    print(json.dumps(result))
    if not result['completed']:raise SystemExit(2)


if __name__=='__main__':main()
