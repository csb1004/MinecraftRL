"""Read inventories from the project's own single-player Minecraft save archives."""
import gzip
import io
import struct
import zipfile
from collections import Counter


def read_nbt(compressed):
    stream=io.BytesIO(gzip.decompress(compressed))
    def number(fmt): return struct.unpack('>'+fmt,stream.read(struct.calcsize('>'+fmt)))[0]
    def string(): return stream.read(number('H')).decode('utf-8')
    def length():
        value=number('i')
        if not 0<=value<=1000000: raise ValueError('invalid NBT collection length')
        return value
    def payload(kind,depth=0):
        if depth>64: raise ValueError('NBT nesting too deep')
        if kind in (1,2,3,4,5,6): return number({1:'b',2:'h',3:'i',4:'q',5:'f',6:'d'}[kind])
        if kind==7:return stream.read(length())
        if kind==8:return string()
        if kind==9:
            subtype=number('B');return [payload(subtype,depth+1) for _ in range(length())]
        if kind==10:
            result={}
            while True:
                subtype=number('B')
                if not subtype:return result
                key=string();result[key]=payload(subtype,depth+1)
        if kind in (11,12):return [number('i' if kind==11 else 'q') for _ in range(length())]
        raise ValueError('unsupported NBT tag')
    kind=number('B');string()
    return payload(kind)


def verify_saved_inventory(archive,expected):
    records=[];level_seen=False;player_seen=False
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            level=name.endswith('/level.dat')
            player='/playerdata/' in name and name.endswith('.dat')
            if not (level or player):continue
            value=read_nbt(bundle.read(name))
            if level:value=value.get('Data',{}).get('Player',{})
            if not isinstance(value.get('Inventory'),list): raise ValueError('saved inventory missing')
            counts=Counter()
            for item in value['Inventory']:
                if item['Count']<0:raise ValueError('negative saved inventory count')
                counts[item['id'].removeprefix('minecraft:')]+=item['Count']
            records.append(dict(file=name,inventory=dict(counts),matches=dict(counts)==expected))
            level_seen|=level;player_seen|=player
    return dict(verified=bool(level_seen and player_seen and all(r['matches'] for r in records)),records=records)
