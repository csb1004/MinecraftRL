"""Windows broker <-> WSL mailbox fixture; no Minecraft or network access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from minecraftrl.recovery_broker import serve_once
from minecraftrl.recovery_draft import validate_draft
from minecraftrl.recovery_exchange import FileRecoveryProvider
from minecraftrl.replanning import make_snapshot
from minecraftrl.runtime import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'results/planning-recovery-mailbox-fixture-20261009'
    exchange=base/'exchange'
    if a.worker:
        observation=dict(server_available=1,error=0,server_tick=200,server_x=10.,server_y=64.,server_z=10.,
            server_inventory={'oak_log':0},server_alive=1,selected_item='air',wood_v3_kind=1,
            wood_v3_target_x=18.,wood_v3_target_y=64.,wood_v3_target_z=10.)
        snapshot=make_snapshot(observation,world_id='synthetic-mailbox-world',episode_id='synthetic-mailbox-episode',
            goal='wood',task=None,generation=1,sequence=200,elapsed_s=25.,status='stalled',
            release_sequence=200,held_key_count=0.)
        packet=dict(request_id='synthetic-mailbox-request',evidence_id='synthetic-stall',
            root_goal=dict(goal='wood',target_count=1),root_reference=snapshot['goal_ref'],
            assessment=dict(classification='stagnation_suspected',replan_candidate=True,goal_completed=False),
            status='stalled',remaining_control_s=155.,runtime_binding=snapshot,
            scope='Explicit synthetic transport fixture, not actual game evidence')
        provider=FileRecoveryProvider(exchange);provider.start(packet);deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            result=provider.poll()
            if result is not None:
                assert result['request_id']==packet['request_id'] and result['choice']=='explore_then_resume'
                assert result['execution_authorized'] is False
                provider.close();write_json(base/'wsl-receipt.json',dict(verified=True,actual_api_calls=0,new_game_actions=0))
                return
            time.sleep(.02)
        provider.close();raise TimeoutError('Cross-OS fixture response timeout')
    base.mkdir(exist_ok=False)
    linux='/mnt/d/DATA/Desktop/MinecraftRL'
    argv=['wsl.exe','-d','Ubuntu-24.04','--cd',linux,'--','env','PYTHONPATH=src',
          'python3','scripts/check_recovery_mailbox_crossos.py','--worker']
    with (base/'wsl-console.txt').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        class Fixture:
            def propose(self,packet):
                return validate_draft(dict(request_id=packet['request_id'],evidence_id=packet['evidence_id'],
                    choice='explore_then_resume',reason='Synthetic Windows broker fixture.'),packet)
        result=serve_once(exchange,lambda:Fixture(),base/'fixture-STOP',wait_timeout_s=35.)
        try:process.wait(timeout=25)
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
        assert process.returncode==0 and result['status']=='delivered'
    receipt=json.loads((base/'wsl-receipt.json').read_text())
    assert receipt['verified'] and (exchange/'closed.json').exists()
    write_json(base/'report.json',dict(verified=True,windows_to_wsl_response=True,wsl_to_windows_request=True,
        actual_api_calls=0,new_game_actions=0,worker_exit_code=process.returncode,scope='Synthetic IPC fixture only'))
    print('Windows/WSL mailbox verified; actual API calls 0; game actions 0')


if __name__=='__main__':main()
