"""Read-only replay of movement evidence; no fixture geometry enters detection."""
import json,math
from pathlib import Path
from minecraftrl.motion_blockage import MotionBlockage
from minecraftrl.runtime import write_json

ROOT=Path(__file__).resolve().parents[1]


def inspect(rows,target):
    detector=MotionBlockage()
    for i,r in enumerate(rows,1):
        evidence=detector.observe(r['server_before'],r['server_after'],r['action'],sequence=i)
        if evidence:
            p=evidence['position'];dx,dz=target[0]-p[0],target[2]-p[2];length=math.hypot(dx,dz)
            if length and (dx*evidence['direction'][0]+dz*evidence['direction'][1])/length>=math.cos(math.radians(40)):
                return dict(first_action=i,evidence=evidence)
    return None


def main():
    cases=[]
    for batch in ('rm-refined-navigation-20261015','exploration-coverage-20261015'):
        for world in ('far','corner'):
            run=ROOT/'results'/batch/('game-'+world)/'run'
            if not (run/'navigation-receipts.json').exists():continue
            rows=json.loads((run/'goal-recovery-rows.json').read_text())
            for r in json.loads((run/'navigation-receipts.json').read_text()):
                window=rows[r['before']['sequence']:r['after']['sequence']]
                cases.append(dict(run=run.relative_to(ROOT).as_posix(),outcome=r['outcome'],steps=len(window),
                    target=r['method']['arguments']['target'],hit=inspect(window,r['method']['arguments']['target'])))
    for run in sorted((ROOT/'results/navigation-refinement-20261014').glob('evaluate-*/run')):
        rows=json.loads((run/'navigation-rows.json').read_text())
        report=json.loads((run/'summary.json').read_text())
        for r in report['navigation_receipts']:
            if not r.get('actions'):continue
            window=[v for v in rows if v['index']==r['index'] and v['owner']=='student']
            cases.append(dict(run=run.relative_to(ROOT).as_posix(),outcome=r['outcome'],steps=len(window),
                target=r['target'],hit=inspect(window,r['target'])))
    result=dict(scope='Saved trajectories only; detection is not proof of recovery or passability',cases=cases,
        completed_segments=sum(c['outcome']=='completed' for c in cases),
        hits_on_completed=sum(c['outcome']=='completed' and c['hit'] is not None for c in cases),
        failed_segments=sum(c['outcome']=='failed' for c in cases),
        hits_on_failed=sum(c['outcome']=='failed' and c['hit'] is not None for c in cases))
    write_json(ROOT/'results/exploration-coverage-20261015/motion-replay.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
