"""Copy a closed diagnostic world and inspect its saved inventory, preserving the original."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('runtime',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];attempt=a.attempt.resolve();runtime=a.runtime.resolve()
    if not attempt.is_relative_to(root/'results') or not runtime.is_relative_to(root/'data/worlds/runtime'):
        raise ValueError('only project-local diagnostic worlds may be preserved')
    summary=json.loads((attempt/'run/summary.json').read_text(encoding='utf-8'))
    assert not summary['process_cleanup']['remaining_live_pids']
    worlds=[p for p in runtime.glob('*/saves/*') if (p/'level.dat').is_file()]
    if len(worlds)!=1:raise ValueError('expected one closed world')
    sys.path.insert(0,str(root/'src'))
    from minecraftrl.saved_inventory import read_nbt
    source=worlds[0];destination=attempt/'world-files-after-shutdown';destination.mkdir(exist_ok=False)
    files=[]
    for path in sorted(source.rglob('*')):
        if path.is_symlink():raise ValueError('world symlink')
        if not path.is_file():continue
        target=destination/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==hashlib.sha256(target.read_bytes()).hexdigest()
        files.append(dict(path=path.relative_to(source).as_posix(),bytes=path.stat().st_size,sha256=sha))
    level=read_nbt((destination/'level.dat').read_bytes())
    inventories={'level.dat/Player':level['Data'].get('Player',{}).get('Inventory',[])}
    for path in (destination/'playerdata').glob('*.dat'):inventories[path.name]=read_nbt(path.read_bytes()).get('Inventory',[])
    result=dict(source=str(source),copy=str(destination),original_preserved=True,files=files,saved_inventory=inventories)
    (attempt/'world-preservation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(files),saved_inventory=inventories)))


if __name__=='__main__':main()
