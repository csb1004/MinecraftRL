"""Post-run fixture geometry diagnosis; NEVER imported by a policy or planner."""
import gzip,json,math,zipfile,zlib
from pathlib import Path
from minecraftrl.saved_inventory import read_nbt
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

ROOT=Path(__file__).resolve().parents[1]


def read_section(archive,cx,cz):
    with zipfile.ZipFile(archive) as z:
        suffix=f'/region/r.{cx//32}.{cz//32}.mca'
        names=[n for n in z.namelist() if n.endswith(suffix) and '/DIM' not in n]
        assert len(names)==1
        region=z.read(names[0]);index=cx%32+(cz%32)*32
        offset=int.from_bytes(region[index*4:index*4+3],'big')*4096
        size=int.from_bytes(region[offset:offset+4],'big');kind=region[offset+4]
        assert offset>=8192 and kind==2
        chunk=read_nbt(gzip.compress(zlib.decompress(region[offset+5:offset+4+size])))
        assert chunk['DataVersion']==2586
        sections=[s for s in chunk['Level']['Sections'] if s['Y']==12];assert len(sections)==1
        return sections[0]


def block(section,x,y,z):
    palette=section['Palette'];bits=max(4,(len(palette)-1).bit_length());per=64//bits
    index=x+16*z+256*(y-192)
    word=int(section['BlockStates'][index//per]) & ((1<<64)-1)
    return palette[(word>>((index%per)*bits))&((1<<bits)-1)]['Name']


def main():
    cases=[]
    for batch in ('rm-refined-navigation-20261015','exploration-coverage-20261015','exploration-motion-20261015'):
        for world in ('far','corner'):
            run=ROOT/'results'/batch/('game-'+world)/'run';path=run/'navigation-receipts.json'
            if not path.exists():continue
            manifest=json.loads((ROOT/'results/blind-recovery-20261012/worlds'/(world+'.json')).read_text())
            archive=ROOT/'results/blind-recovery-20261012/worlds'/(world+'.zip')
            assert digest(archive)==manifest['sha256']
            cx,cz=manifest['chunk'];section=read_section(archive,cx,cz)
            # Verify every enclosing block from the actual archived NBT, not just generator code.
            for x in range(16):
                for z in range(16):
                    if x in (0,15) or z in (0,15):
                        assert all(block(section,x,y,z)=='minecraft:bedrock' for y in range(201,205))
            for index,r in enumerate(json.loads(path.read_text()),1):
                x,y,z=r['method']['arguments']['target'];lo_x,hi_x=cx*16+1.3,cx*16+14.7
                lo_z,hi_z=cz*16+1.3,cz*16+14.7
                gap=math.hypot(max(lo_x-x,0,x-hi_x),max(lo_z-z,0,z-hi_z))
                cases.append(dict(batch=batch,world=world,attempt=index,target=[x,y,z],outcome=r['outcome'],
                    minimum_distance_from_enclosure_interior=gap,arrival_circle_misses_interior=gap>.35+1e-6,
                    player_half_width_assumption=.3,arrival_radius=.35,world_sha256=manifest['sha256'],
                    scope='Necessary enclosure-interior condition only; inside does not imply a clear path'))
    result=dict(evaluator_only=True,supplied_to_policy=False,geometry='Archived NBT enclosing bedrock walls y201..204 verified',cases=cases)
    out=ROOT/'docs/reports/2026-10-15/exploration-target-geometry.json';write_json(out,result)
    print(json.dumps([dict(batch=c['batch'],world=c['world'],attempt=c['attempt'],outside=c['arrival_circle_misses_interior'],gap=round(c['minimum_distance_from_enclosure_interior'],4)) for c in cases]))


if __name__=='__main__':main()
