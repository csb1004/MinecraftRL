"""Compare independent pixel readings with saved server evidence; never repair OCR."""
import argparse
import json
from pathlib import Path

from minecraftrl.f3_coordinates import parse_f3_xyz
from minecraftrl.spatial_memory import SpatialMemory
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];probe=a.probe.resolve();out=a.output.resolve()
    if not all(path.is_relative_to(root/'results') for path in (probe,out)):raise ValueError('project results required')
    out.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((probe/name).read_text(encoding='utf-8'))
    rows={r['stage']:r for r in read('observations.json')}
    world=root/'results/diverse-teacher-worlds-20261008/development-corner.zip'
    m=SpatialMemory(digest(world),str(probe),source='f3_ocr',dimension='overworld')
    checks=[];bound_files=['observations.json','work.json','release.json','capture.json']
    for stage in ('f3_release','f3_visible'):
        value=read('bitmap-'+stage+'.json');server=rows[stage]['server']
        if not value['accepted'] or value['image_sha256']!=digest(probe/(stage+'.png')):
            raise ValueError('unbound pixel reading')
        # The parser/reader did not receive this server position. It is used only below for comparison.
        actual=[float(server['server_'+axis]) for axis in 'xyz']
        errors=[abs(a-b) for a,b in zip(value['position'],actual)]
        if any(e>t for e,t in zip(errors,(.00051,.000006,.00051))):raise ValueError('pixel/server mismatch')
        m.observe(value['position'],int(server['server_tick']))
        checks.append(dict(stage=stage,pixel_position=value['position'],server_position=actual,
            absolute_error=errors,image_sha256=value['image_sha256']))
        bound_files.extend([stage+'.png','bitmap-'+stage+'.json'])
    closed=read('bitmap-f3_closed.json')
    if closed['accepted'] or closed['image_sha256']!=digest(probe/'f3_closed.png'):
        raise ValueError('closed overlay must not supply coordinates')
    bound_files.extend(['f3_closed.png','bitmap-f3_closed.json'])
    rejected=[]
    for name in ('ocr-visible.json','ocr-visible-en.json'):
        value=read(name)
        try:parse_f3_xyz(value['text'])
        except ValueError as exc:rejected.append(dict(file=name,engine=value['engine'],error=str(exc)))
        else:raise ValueError('unexpected generic OCR acceptance; inspect evidence')
        bound_files.append(name)
    work=read('work.json');release=read('release.json');capture=read('capture.json')
    if not work['completed'] or work['cleanup']['remaining_live_pids']:raise ValueError('probe cleanup failed')
    if not release['evidence']['verified'] or release['evidence']['held_key_count']!=0:
        raise ValueError('keys not released')
    m.save(out/'f3-memory.json')
    report=dict(verified=True,pixel_checks=checks,closed_overlay_rejected=True,
        generic_ocr_rejected=rejected,summary=m.summary(),capture=capture,cleanup=work['cleanup'],
        position_source='F3 screenshot pixels only',tick_source='recorded server timing metadata',
        dimension_source='overworld development fixture configuration, not OCR',
        live_policy_navigation_success=False,api_calls=0,optimizer_updates=0,
        scope='Stationary UI coordinate diagnostic; no learned navigation or moving-camera validation',
        source_sha256={name:digest(probe/name) for name in bound_files})
    write_json(out/'report.json',report)
    print(json.dumps(dict(verified=True,frames=len(checks),max_error=max(max(c['absolute_error']) for c in checks),
        closed_rejected=True,generic_ocr_rejections=len(rejected),map_source=m.source)))


if __name__=='__main__':main()
