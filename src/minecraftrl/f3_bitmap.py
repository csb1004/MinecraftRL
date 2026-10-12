"""Strict OCR for the pinned vanilla bitmap font; no correction from game telemetry."""
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
from .f3_coordinates import parse_f3_xyz
from .world_save_patch import UPSTREAM_SHA256
from .world_store import digest


def load_font(jar):
    if digest(jar)!=UPSTREAM_SHA256:raise ValueError('unverified font artifact')
    with zipfile.ZipFile(jar) as archive:
        meta=archive.read('assets/minecraft/font/default.json')
        data=archive.read('assets/minecraft/textures/font/ascii.png')
    provider=next(p for p in json.loads(meta)['providers'] if p.get('file')=='minecraft:font/ascii.png')
    bitmap=np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))[:,:,3]>0
    if bitmap.shape!=(128,128) or provider['ascent']!=7:raise ValueError('unsupported font geometry')
    chars=''.join(provider['chars']);glyphs={}
    for c in 'XYZ:0123456789./-+':
        i=chars.index(c);glyph=bitmap[(i//16)*8:(i//16+1)*8,(i%16)*8:(i%16+1)*8]
        width=int(np.where(glyph.any(axis=0))[0][-1])+1
        glyphs[c]=np.pad(glyph[:,:width],((0,0),(0,1)))
    glyphs[' ']=np.zeros((8,4),dtype=bool)
    return dict(glyphs=glyphs,font_sha256=hashlib.sha256(data).hexdigest(),jar_sha256=UPSTREAM_SHA256)


def read_f3_bitmap(image,font):
    pixels=np.asarray(image)
    if pixels.ndim!=3 or pixels.shape[2]!=3 or pixels.dtype!=np.uint8 or max(pixels.shape[:2])>4096:
        raise ValueError('unsupported F3 image')
    # Native gray debug foreground. Shadows and scene colors are not substituted.
    mask=(pixels[:,:,0]>=210)&(pixels[:,:,0]<=230)&(pixels[:,:,0]==pixels[:,:,1])&(pixels[:,:,1]==pixels[:,:,2])
    glyphs=font['glyphs'];prefix=np.concatenate([glyphs[c] for c in 'XYZ: '],axis=1)
    hits=[]
    for scale in (1,2,3,4):
        scaled=np.repeat(np.repeat(prefix,scale,axis=0),scale,axis=1)
        height,width=scaled.shape;x=2*scale
        if x+width>mask.shape[1]:continue
        for y in range(mask.shape[0]-height+1):
            if np.array_equal(mask[y:y+height,x:x+width],scaled):hits.append((x,y,scale,width))
    if len(hits)!=1:raise ValueError('one exact visible XYZ prefix required')
    x,y,scale,width=hits[0];cursor=x+width;text='XYZ: ';height=8*scale
    scaled={c:np.repeat(np.repeat(g,scale,axis=0),scale,axis=1) for c,g in glyphs.items() if c not in 'XYZ:'}
    for _ in range(100):
        # Two blank spaces delimit the end; a malformed glyph is not silently truncated.
        if not mask[y:y+height,cursor:cursor+8*scale].any():break
        matches=[(c,g) for c,g in scaled.items() if np.array_equal(mask[y:y+height,cursor:cursor+g.shape[1]],g)]
        if len(matches)!=1:raise ValueError('unreadable or ambiguous F3 glyph')
        c,g=matches[0];text+=c;cursor+=g.shape[1]
    else:raise ValueError('F3 line too long')
    position=parse_f3_xyz(text.strip())
    return dict(source='f3_ocr',engine='pinned_vanilla_bitmap_exact',position=position,text=text.strip(),
        render_scale=scale,line_box=[x,y,cursor,y+height],font_sha256=font['font_sha256'],
        font_jar_sha256=font['jar_sha256'],scope='pixel-derived coordinates; no telemetry or language-model correction')
