"""Prepare an immutable development start with a real sightline above the existing wall.

Only saved player pose/motion is changed. No terrain, inventory, policy state, or runtime action.
The ordinary server ray must discover the tree during the subsequent real game.
"""
import gzip
import io
import json
import math
from pathlib import Path
import sys
import zipfile


def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/local-navigation-detour-20261019'
    sys.path[:0]=[str(root/'src'),str(out/'fixture-deps')]
    import nbtlib as n
    from minecraftrl.world_store import digest,validate_archive
    source=root/'results/local-wall-diagnosis-20261017/assets/results/blind-recovery-20261012/worlds/corner.zip'
    output=out/'worlds/known-corner.zip';output.parent.mkdir(exist_ok=True)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    original_sha=digest(source);manifest=json.loads(source.with_suffix('.json').read_text())
    assert manifest['sha256']==original_sha
    position=[227.5,201.,-788.5];dx,dz=8.,1.
    yaw=math.degrees(math.atan2(-dx,dz));pitch=-math.degrees(math.atan2(1.88,math.hypot(dx,dz)))
    def modify(raw,level=False):
        tag=n.File.parse(io.BytesIO(gzip.decompress(raw)));player=tag['Data']['Player'] if level else tag
        assert len(player['Inventory'])==0
        player['Pos']=n.List[n.Double](position);player['Motion']=n.List[n.Double]([0.,0.,0.])
        player['Rotation']=n.List[n.Float]([yaw,pitch]);player['OnGround']=n.Byte(1)
        stream=io.BytesIO();tag.write(stream);return gzip.compress(stream.getvalue(),mtime=0)
    changed=[]
    with zipfile.ZipFile(source) as src,zipfile.ZipFile(output,'x') as dst:
        for entry in src.infolist():
            data=src.read(entry)
            if entry.filename.endswith('/level.dat'):data=modify(data,level=True);changed.append(entry.filename)
            elif '/playerdata/' in entry.filename and entry.filename.endswith('.dat'):
                data=modify(data);changed.append(entry.filename)
            dst.writestr(entry,data)
    validate_archive(output);assert digest(source)==original_sha
    with zipfile.ZipFile(source) as src,zipfile.ZipFile(output) as dst:
        assert all(src.read(name)==dst.read(name) for name in src.namelist() if name not in changed)
    for key in ('pose_source','start_pose','reconstruction'):manifest.pop(key,None)
    manifest.update(sha256=digest(output),source_sha256=original_sha,source=str(source),position=position,
        start_pose=dict(yaw=yaw,pitch=pitch),experimental_split='development',changed_entries=changed,
        scope='Pose-only fixture; identical arena and empty inventory; no policy success credit',
        diagnostic='Visible upper log across the existing wall; real ray acquisition and actual detour must be observed')
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    protocol=dict(scope='Additional development test of discovered-target detouring, not a final evaluation',
        reason='First improved run needed only one approach action; a wall detour after discovery was not exercised',
        cases=['known-baseline-01','known-candidate-01'],maximum_games=2,control_limit_s=180,
        common_start=manifest,sampling_seed=14031,training_updates=0,paid_api_calls=0,
        acceptance='Actual discovered target, learned approach steps around wall, server inventory gain, STOP and PID cleanup')
    (out/'known-target-protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__':main()
