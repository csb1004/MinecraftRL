"""Read a block from a 1.16.5 Anvil snapshot; never reads a live world."""
import gzip
import zlib
import zipfile
from .saved_inventory import read_nbt


def section_block(section,x,y,z):
    palette=section['Palette'];bits=max(4,(len(palette)-1).bit_length())
    if len(palette)==1:return palette[0]['Name']
    index=(y%16)*256+(z%16)*16+x%16;per_long=64//bits
    word=section['BlockStates'][index//per_long] & ((1<<64)-1)
    entry=(word >> ((index%per_long)*bits)) & ((1<<bits)-1)
    return palette[entry]['Name']


def saved_block(archive,position):
    x,y,z=position;cx=x//16;cz=z//16
    suffix='/region/r.'+str(cx//32)+'.'+str(cz//32)+'.mca'
    with zipfile.ZipFile(archive) as bundle:
        names=[n for n in bundle.namelist() if n.endswith(suffix) and '/DIM' not in n]
        if len(names)!=1:raise ValueError('missing or ambiguous overworld region')
        region=bundle.read(names[0])
    offset=4*((cx%32)+(cz%32)*32);sector=int.from_bytes(region[offset:offset+3],'big')
    if not sector:raise ValueError('missing saved chunk')
    start=sector*4096;length=int.from_bytes(region[start:start+4],'big');kind=region[start+4]
    payload=region[start+5:start+4+length]
    if kind==2:payload=gzip.compress(zlib.decompress(payload))
    elif kind!=1:raise ValueError('unsupported chunk compression')
    chunk=read_nbt(payload)
    if chunk['DataVersion']!=2586:raise ValueError('block decoder requires Minecraft 1.16.5')
    level=chunk['Level']
    if (level['xPos'],level['zPos'])!=(cx,cz):raise ValueError('chunk coordinates mismatch')
    sections=[s for s in level['Sections'] if s['Y']==y//16]
    if len(sections)!=1:raise ValueError('missing block section')
    return section_block(sections[0],x,y,z)
