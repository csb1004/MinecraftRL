"""Immutable local MineRL world generations with an atomic current pointer.

Only commit after the environment has acknowledged normal save-and-quit.
Loading never silently falls back from a corrupt generation to a new world.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
import zipfile

from .memory import atomic_json, file_lock
from .runtime import inventory_counts, jsonable
from .tasks import TaskState


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def validate_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=set(); world=None; level=False; total=0
        for entry in archive.infolist():
            parts=entry.filename.split('/')
            if (len(parts)<4 or parts[0]!='.' or parts[1]!='saves' or
                not re.fullmatch(r'[A-Za-z0-9_-]+',parts[2]) or
                any(p in ('..','.') or '\\' in p or ':' in p for p in parts[3:]) or
                any(not p for p in parts[3:-1]) or entry.filename in names or
                (entry.external_attr>>16)&0o170000==0o120000):
                raise ValueError('unsafe_world_archive')
            if world is not None and world!=parts[2]: raise ValueError('multiple_archive_worlds')
            world=parts[2]; names.add(entry.filename); total+=entry.file_size
            if parts[3:]==['level.dat'] and entry.file_size>0: level=True
        if not level or total>8*1024**3: raise ValueError('invalid_world_archive')
        if archive.testzip() is not None: raise ValueError('world_archive_crc_error')


def observation_evidence(observation):
    location=jsonable(observation.get('location_stats',{}))
    for key in ('xpos','ypos','zpos'):
        value=location.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('world_location_unavailable')
    return {'inventory':{k:v for k,v in inventory_counts(observation).items() if v},
            'location_stats':location,'life_stats':jsonable(observation.get('life_stats',{}))}


def validate_restored_observation(bundle,observation):
    actual=observation_evidence(observation); expected=bundle['observation']
    if actual['inventory']!=expected['inventory']: raise ValueError('restored_inventory_mismatch')
    # MineRL advances initial neutral frames. Bound the resulting position drift;
    # this is a resume check, not a claim of bit-exact physics continuation.
    distance=math.sqrt(sum((actual['location_stats'][k]-expected['location_stats'][k])**2
                           for k in ('xpos','ypos','zpos')))
    if distance>2.: raise ValueError('restored_position_mismatch')
    return {'inventory_match':True,'position_distance_blocks':distance,
            'position_tolerance_blocks':2.,'actual':actual}


class WorldStore:
    def __init__(self,root,slot):
        if not isinstance(slot,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',slot):
            raise ValueError('invalid_world_slot')
        root=Path(root).resolve(); self.folder=root/slot
        if self.folder.is_symlink(): raise ValueError('world_slot_symlink')
        self.folder.mkdir(parents=True,exist_ok=True); self.lock=None

    def __enter__(self):
        self.lock=file_lock(self.folder/'writer.lock'); self.lock.__enter__(); return self

    def __exit__(self,*exc): return self.lock.__exit__(*exc)

    def load(self):
        pointer=self.folder/'current.json'
        if not pointer.exists(): return None
        current=json.loads(pointer.read_text(encoding='utf-8')); generation=current.get('generation')
        if not isinstance(generation,str) or not re.fullmatch(r'[0-9a-f]{32}',generation):
            raise ValueError('invalid_world_generation')
        folder=self.folder/generation
        if folder.is_symlink(): raise ValueError('world_generation_symlink')
        manifest=folder/'manifest.json'; archive=folder/'world.zip'
        if manifest.is_symlink() or archive.is_symlink(): raise ValueError('world_file_symlink')
        if digest(manifest)!=current['manifest_sha256']: raise ValueError('world_manifest_hash_mismatch')
        bundle=json.loads(manifest.read_text(encoding='utf-8'))
        if bundle['schema']!=1 or digest(archive)!=bundle['archive_sha256']:
            raise ValueError('world_archive_hash_mismatch')
        validate_archive(archive)
        TaskState.restore(bundle['task_checkpoint'],bundle['world_id'])
        bundle['archive']=str(archive); bundle['generation']=generation
        return bundle

    def commit(self,world,task_checkpoint,observation):
        world=Path(world)
        if world.is_symlink() or not (world/'level.dat').is_file(): raise ValueError('missing_saved_world')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',world.name): raise ValueError('invalid_world_name')
        TaskState.restore(task_checkpoint,task_checkpoint['world_id'])
        evidence=observation_evidence(observation)
        generation=uuid.uuid4().hex; destination=self.folder/generation; destination.mkdir()
        archive_path=destination/'world.zip'
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in sorted(world.rglob('*')):
                if path.is_symlink(): raise ValueError('saved_world_symlink')
                if path.is_file() and path.name!='session.lock':
                    # ZipFile.write strips './', required by upstream's loader.
                    info=zipfile.ZipInfo('./saves/'+world.name+'/'+path.relative_to(world).as_posix())
                    info.compress_type=zipfile.ZIP_DEFLATED
                    with path.open('rb') as source, archive.open(info,'w',force_zip64=True) as target:
                        shutil.copyfileobj(source,target,1024*1024)
        with archive_path.open('r+b') as stream: stream.flush(); os.fsync(stream.fileno())
        validate_archive(archive_path)
        bundle={'schema':1,'world_id':task_checkpoint['world_id'],'task_checkpoint':task_checkpoint,
                'archive_sha256':digest(archive_path),'observation':evidence,'save_method':'minerl_quit_acknowledged'}
        atomic_json(destination/'manifest.json',bundle)
        atomic_json(self.folder/'current.json',{'generation':generation,'manifest_sha256':digest(destination/'manifest.json')})
        return destination
