"""Evidence for bounded passive pickups while resuming a saved world, never policy input."""
from collections import Counter
import gzip
import math
import zipfile
import zlib

from .saved_inventory import read_nbt, verify_saved_inventory


def eligible_drop_position(player_position,item,*,include_falling=False):
    pos=item.get('Pos',[])
    if len(pos)!=3 or not all(math.isfinite(v) for v in pos):return False
    if math.dist(player_position,pos)<=3.:return True
    motion=item.get('Motion',[])
    return bool(include_falling and not item.get('NoGravity',False) and len(motion)==3
        and all(math.isfinite(v) for v in motion) and math.hypot(motion[0],motion[2])<=.25
        and -.5<=motion[1]<=.25 and math.hypot(pos[0]-player_position[0],pos[2]-player_position[2])<=1.5
        and 0<=pos[1]-player_position[1]<=6.)


def saved_drop_budget(archive, expected_inventory,*,include_falling=False):
    if not verify_saved_inventory(archive, expected_inventory)['verified']:
        raise ValueError('saved_inventory_differs_from_manifest')
    records=[]; budget=Counter(); seen=set()
    with zipfile.ZipFile(archive) as bundle:
        levels=[n for n in bundle.namelist() if n.endswith('/level.dat')]
        if len(levels)!=1: raise ValueError('ambiguous_saved_player')
        player=read_nbt(bundle.read(levels[0]))['Data']['Player']; position=player['Pos']
        if len(position)!=3 or not all(math.isfinite(v) for v in position): raise ValueError('invalid_saved_position')
        if player.get('Dimension','minecraft:overworld')!='minecraft:overworld':
            raise ValueError('startup_pickups_require_overworld')
        # This is a conservative nearby-item budget, not proof of the pickup's exact UUID.
        radius=3.; regions={}
        for cx in range(math.floor((position[0]-radius)/16),math.floor((position[0]+radius)/16)+1):
            for cz in range(math.floor((position[2]-radius)/16),math.floor((position[2]+radius)/16)+1):
                suffix='/region/r.%s.%s.mca'%(cx//32,cz//32)
                names=[n for n in bundle.namelist() if n.endswith(suffix) and '/DIM' not in n]
                if not names: continue
                if len(names)!=1: raise ValueError('ambiguous_saved_region')
                name=names[0]
                if name not in regions: regions[name]=bundle.read(name)
                raw=regions[name]; index=4*((cx%32)+(cz%32)*32)
                if len(raw)<8192: raise ValueError('invalid_saved_region')
                offset=int.from_bytes(raw[index:index+3],'big')*4096
                if not offset: continue
                if offset<8192 or offset+5>len(raw): raise ValueError('invalid_chunk_offset')
                size=int.from_bytes(raw[offset:offset+4],'big');kind=raw[offset+4]
                if size<1 or offset+4+size>len(raw): raise ValueError('invalid_chunk_size')
                payload=raw[offset+5:offset+4+size]
                if kind==2: payload=gzip.compress(zlib.decompress(payload))
                elif kind!=1: raise ValueError('unsupported_chunk_compression')
                chunk=read_nbt(payload)
                if chunk.get('DataVersion')!=2586: raise ValueError('startup_pickups_require_1_16_5')
                level=chunk['Level']
                if (level['xPos'],level['zPos'])!=(cx,cz): raise ValueError('chunk_coordinates_mismatch')
                for item in level.get('Entities',[]):
                    if item.get('id')!='minecraft:item': continue
                    pos=item.get('Pos',[]); data=item.get('Item',{}); count=data.get('Count',0)
                    if len(pos)!=3 or not all(math.isfinite(v) for v in pos): continue
                    distance=math.dist(position,pos)
                    if (not eligible_drop_position(position,item,include_falling=include_falling) or not 0<=item.get('PickupDelay',32767)<=60
                            or not 0<=item.get('Age',6000)<5940 or item.get('Owner') is not None
                            or type(count) is not int or not 0<count<=64): continue
                    identity=tuple(item.get('UUID',[]))
                    if len(identity)!=4 or identity in seen: raise ValueError('invalid_item_identity')
                    seen.add(identity)
                    item_id=data.get('id','')
                    if not item_id.startswith('minecraft:'): raise ValueError('unsupported_saved_item')
                    key=item_id.removeprefix('minecraft:'); budget[key]+=count
                    records.append(dict(item=key,count=count,uuid=list(identity),distance_blocks=distance,
                                        pickup_delay=item['PickupDelay'],age=item['Age'],position=pos,motion=item.get('Motion')))
    return dict(inventory_budget=dict(budget),nearby_items=records,radius_blocks=radius,
                falling_column={'horizontal_radius':1.5,'above_player':6.} if include_falling else None,
                scope='consistent with saved nearby drops; exact consumed UUID not observed',policy_action_credit=False)


def validate_pickup_delta(expected, actual, evidence):
    delta={k:actual.get(k,0)-expected.get(k,0) for k in set(expected)|set(actual)
           if actual.get(k,0)!=expected.get(k,0)}
    if not delta or any(v<=0 or v>evidence['inventory_budget'].get(k,0) for k,v in delta.items()):
        raise ValueError('restored_inventory_mismatch_not_explained_by_saved_drops')
    return dict(evidence,inventory_delta=delta,consistent_with_saved_drops=True)
