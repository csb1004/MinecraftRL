"""Offline 1.16.5 arena copies. Requires isolated nbtlib; never edits a live save."""
import gzip,io,json,math,zipfile,zlib
from pathlib import Path
import nbtlib as n
from .world_store import digest,validate_archive
from .saved_inventory import read_nbt


def signed(value):return value if value<2**63 else value-2**64


def section(y,blocks):
    if len(blocks)!=4096:raise ValueError('section must have 4096 blocks')
    palette=list(dict.fromkeys(blocks));lookup={name:i for i,name in enumerate(palette)}
    bits=max(4,(len(palette)-1).bit_length());per=64//bits;words=[0]*math.ceil(4096/per)
    for i,name in enumerate(blocks):words[i//per]|=lookup[name]<<((i%per)*bits)
    entries=[]
    for name in palette:
        tag=n.Compound(Name=n.String(name))
        if name.endswith('_log'):tag['Properties']=n.Compound(axis=n.String('y'))
        if name.endswith('_leaves'):tag['Properties']=n.Compound(distance=n.String('1'),persistent=n.String('true'))
        entries.append(tag)
    return n.Compound(Y=n.Byte(y),Palette=n.List[n.Compound](entries),
                      BlockStates=n.LongArray([signed(v) for v in words]),
                      SkyLight=n.ByteArray([-1]*2048),BlockLight=n.ByteArray([0]*2048))


def chunk_bytes(region,index):
    sector=int.from_bytes(region[index*4:index*4+3],'big')
    if not sector:raise ValueError('source chunk missing')
    offset=sector*4096;size=int.from_bytes(region[offset:offset+4],'big');kind=region[offset+4]
    body=region[offset+5:offset+4+size]
    if kind==2:return zlib.decompress(body)
    if kind==1:return gzip.decompress(body)
    raise ValueError('unsupported compression')


def replace_chunk(region,index,raw):
    if len(region)<8192 or not 0<=index<1024:raise ValueError('invalid region sectors')
    for i in range(1024):
        offset=int.from_bytes(region[i*4:i*4+3],'big')*4096
        if not offset:continue
        if offset<8192 or offset+5>len(region):raise ValueError('missing chunk header')
        size=int.from_bytes(region[offset:offset+4],'big')
        if size<1 or offset+4+size>len(region):raise ValueError('truncated chunk payload')
    region=region.ljust(math.ceil(len(region)/4096)*4096,b'\0')
    body=zlib.compress(raw);packet=(len(body)+1).to_bytes(4,'big')+b'\2'+body
    count=math.ceil(len(packet)/4096)
    if count>255:raise ValueError('chunk too large')
    result=bytearray(region);sector=len(result)//4096
    result[index*4:index*4+4]=sector.to_bytes(3,'big')+bytes([count])
    result.extend(packet.ljust(count*4096,b'\0'))
    return bytes(result)


def arena(kind,*,wall_end=None,mirror_z=False):
    if kind not in ('break','detour','death','recovery'):raise ValueError('unknown arena')
    if type(mirror_z) is not bool:raise ValueError('mirror must be boolean')
    if (wall_end is not None or mirror_z) and kind!='recovery':raise ValueError('curriculum requires recovery arena')
    if wall_end is not None and (type(wall_end) is not int or not 3<=wall_end<=13):raise ValueError('invalid wall end')
    blocks={(x,200,z):'minecraft:bedrock' for x in range(16) for z in range(16)}
    for x in range(16):
        for z in range(16):
            if x in (0,15) or z in (0,15):
                for y in range(201,205):blocks[x,y,z]='minecraft:bedrock'
    if kind!='death':
        end=wall_end if wall_end is not None else 13
        for z in (range(1,end+1) if kind in ('break','recovery') else (3,)):
            for y in (201,202):blocks[7,y,z]=('minecraft:dirt' if kind=='break' else
                                            'minecraft:bedrock' if kind=='recovery' else 'minecraft:stone')
        for y in range(201,205):blocks[11,y,3]='minecraft:oak_log'
        for x in range(10,13):
            for z in range(2,5):blocks[x,205,z]='minecraft:oak_leaves'
    return {(x,y,15-z):v for (x,y,z),v in blocks.items()} if mirror_z else blocks


def encode(tag):
    stream=io.BytesIO();tag.write(stream);return stream.getvalue()


def start_pose(blocks,*,start_xz=None,mirror_z=False,start_yaw=None):
    x,z=(3.5,3.5) if start_xz is None else start_xz
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 1.3<=v<=14.7 for v in (x,z)):
        raise ValueError('invalid arena start coordinates')
    if mirror_z:z=16-z
    for bx in range(math.floor(x-.3+1e-7),math.floor(x+.3-1e-7)+1):
        for bz in range(math.floor(z-.3+1e-7),math.floor(z+.3-1e-7)+1):
            if (bx,200,bz) not in blocks or any((bx,y,bz) in blocks for y in (201,202)):
                raise ValueError('arena start collides or lacks floor')
    yaw=-90. if start_xz is None else math.degrees(math.atan2(-(11.5-x),(12.5 if mirror_z else 3.5)-z))
    if start_yaw is not None:
        if type(start_yaw) not in (int,float) or not math.isfinite(start_yaw) or not -180<=start_yaw<=180:
            raise ValueError('invalid canonical start yaw')
        yaw=180-start_yaw if mirror_z else start_yaw
    return [float(x),201.,float(z)],yaw


def make_arena(source,output,kind,*,wall_end=None,mirror_z=False,start_xz=None,start_yaw=None):
    if (start_xz is not None or start_yaw is not None) and kind!='recovery':raise ValueError('custom start requires recovery arena')
    source=Path(source);output=Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    validate_archive(source);source_sha=digest(source);blocks=arena(kind,wall_end=wall_end,mirror_z=mirror_z)
    manifest=json.loads(source.with_suffix('.json').read_text())
    if manifest['sha256']!=source_sha:raise ValueError('source archive hash mismatch')
    with zipfile.ZipFile(source) as original:
        level_name=next(x for x in original.namelist() if x.endswith('/level.dat'))
        level=n.File.parse(io.BytesIO(gzip.decompress(original.read(level_name))))
        old=level['Data']['Player']['Pos'];cx=math.floor(float(old[0]))//16;cz=math.floor(float(old[2]))//16
        suffix=f'/region/r.{cx//32}.{cz//32}.mca'
        region_name=next(x for x in original.namelist() if x.endswith(suffix) and '/DIM' not in x)
        region=original.read(region_name);index=cx%32+(cz%32)*32
        chunk=n.File.parse(io.BytesIO(chunk_bytes(region,index)))
        if int(chunk['DataVersion'])!=2586:raise ValueError('requires 1.16.5')
        data=chunk['Level']
        # Upper chunk replacement is explicit; preserve lower natural terrain.
        data['Sections']=n.List[n.Compound]([s for s in data['Sections'] if int(s['Y'])<12]+
            [section(12,[blocks.get((i%16,192+i//256,(i//16)%16),'minecraft:air') for i in range(4096)])])
        data['Entities']=n.List[n.Compound]([]);data['TileEntities']=n.List[n.Compound]([])
        data['TileTicks']=n.List[n.Compound]([]);data['LiquidTicks']=n.List[n.Compound]([])
        data['isLightOn']=n.Byte(0)
        heights=[max(y for (bx,y,bz) in blocks if bx==x and bz==z)+1 for z in range(16) for x in range(16)]
        packed=[0]*37
        for i,value in enumerate(heights):packed[i//7]|=value<<((i%7)*9)
        for key in data['Heightmaps']:data['Heightmaps'][key]=n.LongArray([signed(v) for v in packed])
        replacements={region_name:replace_chunk(region,index,encode(chunk))}
        relative,yaw=start_pose(blocks,start_xz=start_xz,mirror_z=mirror_z,start_yaw=start_yaw)
        position=[cx*16+relative[0],relative[1],cz*16+relative[2]]
        def player(p):
            p['Pos']=n.List[n.Double](position);p['Motion']=n.List[n.Double]([0.,0.,0.])
            p['Rotation']=n.List[n.Float]([yaw,-13.]);p['Inventory']=n.List[n.Compound]([])
            p['Health']=n.Float(20.);p['foodLevel']=n.Int(20);p['foodSaturationLevel']=n.Float(5.)
            p['FallDistance']=n.Float(0.);p['Fire']=n.Short(32767 if kind=='death' else -20)
            p['DeathTime']=n.Short(0);p['HurtTime']=n.Short(0);p['Air']=n.Short(300)
            p['OnGround']=n.Byte(1);p['ActiveEffects']=n.List[n.Compound]([])
            p['playerGameType']=n.Int(0)
        player(level['Data']['Player'])
        level['Data']['GameRules']['doMobSpawning']=n.String('false')
        level['Data']['GameRules']['doDaylightCycle']=n.String('false')
        level['Data']['DayTime']=n.Long(6000)
        replacements[level_name]=gzip.compress(encode(level),mtime=0)
        for name in original.namelist():
            if '/playerdata/' in name and name.endswith('.dat'):
                p=n.File.parse(io.BytesIO(gzip.decompress(original.read(name))));player(p)
                replacements[name]=gzip.compress(encode(p),mtime=0)
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as target:
            for entry in original.infolist():target.writestr(entry,replacements.get(entry.filename,original.read(entry)))
    validate_archive(output)
    assert digest(source)==source_sha
    result=dict(seed=manifest['seed'],sha256=digest(output),source_sha256=source_sha,source=str(source),
                kind=kind,position=position,chunk=[cx,cz],target=[cx*16+11,201,cz*16+(12 if mirror_z else 3)],
                barrier_material='dirt' if kind=='break' else 'stone' if kind=='detour' else 'bedrock' if kind=='recovery' else None,
                scope='offline synthetic arena; no policy actions or acquisition credit in construction')
    if wall_end is not None or mirror_z:result['curriculum']=dict(wall_end=13 if wall_end is None else wall_end,mirror_z=mirror_z)
    if start_xz is not None or start_yaw is not None:result['start_pose']=dict(canonical_xz=list(start_xz or (3.5,3.5)),relative_position=relative,yaw=yaw,pitch=-13.)
    output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    from .saved_block import saved_block
    for (x,y,z),name in blocks.items():assert saved_block(output,(cx*16+x,y,cz*16+z))==name
    return result
