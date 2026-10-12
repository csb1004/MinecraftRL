"""Read-only video storyboard for manual demonstration review (no training)."""
import argparse
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('output',type=Path)
p.add_argument('--frames',type=int,nargs='+',required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
v=cv2.VideoCapture(str(a.video));canvas=np.zeros((((len(a.frames)+3)//4)*205,4*320,3),dtype=np.uint8)
try:
    for slot,index in enumerate(a.frames):
        v.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=v.read()
        if not ok:raise ValueError(f'frame unavailable: {index}')
        y,x=divmod(slot,4);y*=205;x*=320
        canvas[y+25:y+205,x:x+320]=cv2.resize(frame,(320,180))
        cv2.putText(canvas,str(index),(x+5,y+19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    if not cv2.imwrite(str(a.output),canvas):raise IOError('storyboard write failed')
finally:v.release()
