"""Inspect observed maps and edit named places in a new copy; never send game input."""
import argparse
import json
from pathlib import Path
from minecraftrl.spatial_memory import SpatialMemory


def main():
    p=argparse.ArgumentParser();p.add_argument('map',type=Path)
    p.add_argument('--world-id',required=True);p.add_argument('--dimension',default='overworld')
    p.add_argument('--source',choices=['privileged_server','f3_ocr'],required=True)
    op=p.add_mutually_exclusive_group();op.add_argument('--mark');op.add_argument('--remove');op.add_argument('--route-to')
    p.add_argument('--cell',type=int,nargs=3);p.add_argument('--start-cell',type=int,nargs=3)
    p.add_argument('--mode',choices=['explore','return','resource'],default='return');p.add_argument('--output',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for path in [a.map]+([a.output] if a.output else []):
        if not path.resolve().is_relative_to(root/'results') and not path.resolve().is_relative_to(root/'data'):
            raise ValueError('map paths must remain in project data or results')
    m=SpatialMemory.load(a.map,world_id=a.world_id,source=a.source,dimension=a.dimension)
    if a.mark or a.remove:
        if not a.output or a.output.exists():raise ValueError('edits require a new output copy')
        if a.mark:
            if a.cell is None:raise ValueError('--mark requires observed --cell X Y Z')
            m.set_landmark(a.mark,a.cell)
        else:m.remove_landmark(a.remove)
        a.output.parent.mkdir(parents=True,exist_ok=True);m.save(a.output)
    if a.route_to:
        if a.start_cell is None:raise ValueError('--route-to requires --start-cell X Y Z')
        result=m.route(m.landmarks[a.route_to],mode=a.mode,start=a.start_cell)
    else:result=m.summary()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
