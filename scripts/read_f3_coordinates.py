"""Read a saved F3 screenshot locally. No game actions or telemetry input."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from minecraftrl.f3_bitmap import load_font,read_f3_bitmap
from minecraftrl.world_store import digest


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--font-jar',type=Path,default=root/'.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('output must be in project results')
    if out.exists():raise FileExistsError('preserve existing coordinate evidence')
    result=dict(image=str(a.image.resolve()),image_sha256=digest(a.image),accepted=False)
    try:
        font=load_font(a.font_jar)
        with Image.open(a.image) as screen:
            result.update(read_f3_bitmap(np.asarray(screen.convert('RGB')),font))
        result['accepted']=True
    except ValueError as exc:
        result['error']=str(exc)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    if not result['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
