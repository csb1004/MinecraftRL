"""Offline saved-start fixtures. Layout/coordinates never enter actor inputs."""
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from .world_store import digest, validate_archive
from .runtime import write_json


def geometry(layout):
    if set(layout)!={'item','distance','bearing','view_offset','pitch','obstacle'}:
        raise ValueError('unsupported recovery layout fields')
    if layout['item'] not in ('oak_log','birch_log','dirt') or layout['obstacle'] not in ('none','step','occluder'):
        raise ValueError('unsupported fixture item/obstacle')
    for key,lo,hi in (('distance',2.4,5.5),('bearing',-180,180),('view_offset',-180,180),('pitch',0,50)):
        value=layout[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('invalid fixture '+key)
    blocks={(x,200,z):'minecraft:bedrock' for x in range(16) for z in range(16)}
    for x in range(16):
        for z in range(16):
            if x in (0,15) or z in (0,15):
                for y in (201,202,203):blocks[x,y,z]='minecraft:bedrock'
    # No other harvestable target: dirt gained by mining the floor cannot masquerade as pickup.
    angle=math.radians(layout['bearing']);distance=layout['distance']
    pos=[7.5,201.,7.5]
    item=[7.5+math.sin(angle)*distance,201.05,7.5+math.cos(angle)*distance]
    if layout['obstacle']=='step':
        x,z=math.floor(item[0]),math.floor(item[2])
        blocks[x,201,z]='minecraft:bedrock';item[1]+=1
    if layout['obstacle']=='occluder':
        x,z=math.floor((item[0]+pos[0])/2),math.floor((item[2]+pos[2])/2)
        blocks[x,201,z]=blocks[x,202,z]='minecraft:bedrock'
    yaw=(-layout['bearing']+layout['view_offset']+180)%360-180
    return blocks,pos,item,yaw


def make_fixture(source,output,layout,split):
    import nbtlib as n
    from .rl_fixture import chunk_bytes,replace_chunk,section,encode,signed
    from .saved_inventory import verify_saved_inventory
    if split not in ('train','development'):raise ValueError('explicit development or training fixture required')
    blocks,relative,item_relative,yaw=geometry(layout)
    source,output=Path(source),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    validate_archive(source);source_sha=digest(source)
    meta=json.loads(source.with_suffix('.json').read_text(encoding='utf-8-sig'))
    if meta['sha256']!=source_sha:raise ValueError('source hash mismatch')
    with zipfile.ZipFile(source) as original:
        level_name=next(x for x in original.namelist() if x.endswith('/level.dat'))
        level=n.File.parse(io.BytesIO(gzip.decompress(original.read(level_name))))
        if int(level['Data']['WorldGenSettings']['seed'])!=meta['seed']:raise ValueError('source seed mismatch')
        old=level['Data']['Player']['Pos'];cx,cz=math.floor(float(old[0]))//16,math.floor(float(old[2]))//16
        suffix=f'/region/r.{cx//32}.{cz//32}.mca'
        region_name=next(x for x in original.namelist() if x.endswith(suffix) and '/DIM' not in x)
        region=original.read(region_name);index=cx%32+(cz%32)*32
        chunk=n.File.parse(io.BytesIO(chunk_bytes(region,index)))
        if int(chunk['DataVersion'])!=2586:raise ValueError('requires pinned 1.16.5 format')
        data=chunk['Level']
        data['Sections']=n.List[n.Compound]([s for s in data['Sections'] if int(s['Y'])<12]+
            [section(12,[blocks.get((i%16,192+i//256,(i//16)%16),'minecraft:air') for i in range(4096)])])
        position=[cx*16+relative[0],relative[1],cz*16+relative[2]]
        item_position=[cx*16+item_relative[0],item_relative[1],cz*16+item_relative[2]]
        identity=hashlib.sha256(json.dumps(dict(layout=layout,seed=meta['seed']),sort_keys=True).encode()).digest()[:16]
        entity=n.Compound(id=n.String('minecraft:item'),
            UUID=n.IntArray([int.from_bytes(identity[i:i+4],'big',signed=True) for i in range(0,16,4)]),
            Pos=n.List[n.Double](item_position),Motion=n.List[n.Double]([0.,0.,0.]),
            Rotation=n.List[n.Float]([0.,0.]),Age=n.Short(0),PickupDelay=n.Short(0),Health=n.Short(5),
            OnGround=n.Byte(0),Invulnerable=n.Byte(0),NoGravity=n.Byte(0),
            Item=n.Compound(id=n.String('minecraft:'+layout['item']),Count=n.Byte(1)))
        data['Entities']=n.List[n.Compound]([entity])
        for key in ('TileEntities','TileTicks','LiquidTicks'):data[key]=n.List[n.Compound]([])
        data['isLightOn']=n.Byte(0)
        heights=[max(y for bx,y,bz in blocks if bx==x and bz==z)+1 for z in range(16) for x in range(16)]
        packed=[0]*37
        for i,value in enumerate(heights):packed[i//7]|=value<<((i%7)*9)
        for key in data['Heightmaps']:data['Heightmaps'][key]=n.LongArray([signed(v) for v in packed])
        replacements={region_name:replace_chunk(region,index,encode(chunk))}
        def player(p):
            p['Pos']=n.List[n.Double](position);p['Motion']=n.List[n.Double]([0.,0.,0.])
            p['Rotation']=n.List[n.Float]([yaw,layout['pitch']]);p['Inventory']=n.List[n.Compound]([])
            p['Health']=n.Float(20.);p['foodLevel']=n.Int(20);p['foodSaturationLevel']=n.Float(5.)
            p['FallDistance']=n.Float(0.);p['Fire']=n.Short(-20);p['DeathTime']=n.Short(0)
            p['HurtTime']=n.Short(0);p['Air']=n.Short(300);p['OnGround']=n.Byte(1)
            p['ActiveEffects']=n.List[n.Compound]([]);p['playerGameType']=n.Int(0)
        player(level['Data']['Player'])
        level['Data']['GameRules']['doMobSpawning']=n.String('false')
        level['Data']['GameRules']['doDaylightCycle']=n.String('false')
        level['Data']['GameRules']['randomTickSpeed']=n.String('0')
        level['Data']['DayTime']=n.Long(6000)
        replacements[level_name]=gzip.compress(encode(level),mtime=0)
        for name in original.namelist():
            if '/playerdata/' in name and name.endswith('.dat'):
                tag=n.File.parse(io.BytesIO(gzip.decompress(original.read(name))));player(tag)
                replacements[name]=gzip.compress(encode(tag),mtime=0)
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'x') as target:
            for entry in original.infolist():target.writestr(copy.copy(entry),replacements.get(entry.filename,original.read(entry)))
    validate_archive(output)
    if digest(source)!=source_sha:raise ValueError('source world changed')
    verification=verify_saved_inventory(output,{})
    if not verification['verified']:raise ValueError('fixture inventory is not empty')
    from .saved_block import section_block
    with zipfile.ZipFile(output) as archive:
        checked=n.File.parse(io.BytesIO(chunk_bytes(archive.read(region_name),index)))['Level']
    checked_section=next(s for s in checked['Sections'] if int(s['Y'])==12)
    for (x,y,z),value in blocks.items():
        if section_block(checked_section,x,y,z)!=value:raise ValueError('fixture block mismatch')
    if len(checked['Entities'])!=1 or str(checked['Entities'][0]['Item']['id'])!='minecraft:'+layout['item']:
        raise ValueError('fixture dropped item mismatch')
    result=dict(seed=meta['seed'],sha256=digest(output),source_sha256=source_sha,
        experimental_split=split,layout=layout,position=position,yaw=yaw,pitch=layout['pitch'],
        item_position=item_position,item_uuid=identity.hex(),saved_inventory_verified=True,
        scope='synthetic pickup start, non-harvestable local arena; offline setup only; never actor input',
        random_tick_speed=0,policy_actions=0,performance_samples=0)
    write_json(output.with_suffix('.json'),result)
    return result
