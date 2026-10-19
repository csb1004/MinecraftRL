"""Validate the bounded server camera-ray receipt; never choose or dispatch actions."""
import math


def validate_acquisition(obs):
    e=obs.get('wood_camera_acquisition')
    if e is None:return None
    if (e.get('source')!='privileged_camera_surface_rays_v1' or e.get('tick')!=obs['server_tick']
            or obs.get('wood_camera_rays_enabled')!=1 or obs.get('wood_v3_kind')!=1):
        raise ValueError('unbound visible wood receipt')
    keys=('yaw_offset','pitch_offset','yaw','pitch','range','eye_x','eye_y','eye_z',
          'hit_x','hit_y','hit_z','block_x','block_y','block_z')
    if any(type(e.get(k)) not in (int,float) or not math.isfinite(e[k]) for k in keys):
        raise ValueError('nonfinite camera ray')
    dy,dp=e['yaw_offset'],e['pitch_offset']
    if abs(dy)>40 or abs(dp)>20 or dy%5 or dp%5 or e['range']!=16:
        raise ValueError('camera ray outside configured sampling field')
    eye=[e['eye_'+a] for a in 'xyz'];hit=[e['hit_'+a] for a in 'xyz']
    block=[e['block_'+a] for a in 'xyz']
    if block!=[obs['wood_v3_target_'+a] for a in 'xyz'] or any(int(v)!=v for v in block):
        raise ValueError('ray hit does not bind tracked log')
    if any(not b-1e-4<=h<=b+1.0001 for b,h in zip(block,hit)):
        raise ValueError('ray hit outside target block')
    ray=[h-o for h,o in zip(hit,eye)];length=math.sqrt(sum(v*v for v in ray))
    if not 0<length<=16.0001:raise ValueError('camera ray range')
    direction=[v/length for v in ray]
    def look(p,y):
        p,y=math.radians(p),math.radians(y)
        return [-math.sin(y)*math.cos(p),-math.sin(p),math.cos(y)*math.cos(p)]
    expected=look(max(-89,min(89,e['pitch']+dp)),e['yaw']+dy)
    if sum(a*b for a,b in zip(direction,expected))<.99999:raise ValueError('ray direction mismatch')
    forward=look(e['pitch'],e['yaw']);yaw=math.radians(e['yaw']);pitch=math.radians(e['pitch'])
    right=[math.cos(yaw),0,math.sin(yaw)]
    up=[-math.sin(yaw)*math.sin(pitch),math.cos(pitch),math.cos(yaw)*math.sin(pitch)]
    dot=lambda v:sum(a*b for a,b in zip(direction,v))
    depth=dot(forward)
    if depth<=0 or abs(dot(up)/depth)>math.tan(math.radians(35))+1e-4 or abs(dot(right)/depth)>math.tan(math.radians(35))*640/360+1e-4:
        raise ValueError('ray outside fixed 640x360, 70-degree camera frustum')
    return e
