"""Optical-flow timing probe for official demonstration pairs; CPU only."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from minecraftrl.runtime import write_json

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
results=[]
for source in sorted(a.directory.glob('*.jsonl')):
    rows=[json.loads(x) for x in source.open()];v=cv2.VideoCapture(str(source.with_suffix('.mp4')))
    assert v.isOpened();fps=v.get(cv2.CAP_PROP_FPS);previous=None;flows=[];decoded=0
    try:
        while True:
            ok,frame=v.read()
            if not ok:break
            if decoded<1100:
                gray=cv2.cvtColor(cv2.resize(frame,(160,90)),cv2.COLOR_BGR2GRAY)
                if previous is not None:
                    flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
                    flows.append((decoded-1,float(np.median(flow[15:65,20:140,0]))))
                previous=gray
            decoded+=1
    finally:v.release()
    comparisons={}
    for lag in (-2,-1,0,1,2):
        pairs=[]
        for i,flow in flows:
            j=i+lag
            if not 50<=j<len(rows) or rows[j]['isGuiOpen']:continue
            mouse=rows[j]['mouse'];dx=mouse['dx']*.15;dy=mouse['dy']*.15
            if .15<=abs(dx)<=5 and abs(dy)<=2:pairs.append((-dx,flow))
        corr=float(np.corrcoef(np.asarray(pairs).T)[0,1]) if len(pairs)>3 else None
        comparisons[str(lag)]=dict(samples=len(pairs),correlation=corr)
    results.append(dict(session=source.stem,rows=len(rows),decoded_frames=decoded,fps=fps,comparisons=comparisons))
write_json(a.output,dict(rows=results,scope='Optical horizontal motion versus recorded yaw; empirical timing evidence, not gameplay success'))
print(json.dumps(results))
