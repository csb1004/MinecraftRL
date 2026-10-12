"""Read-only, explicitly selected MyWife user context; never imports its runtime."""
import json
import math
from pathlib import Path


def load_user_context(path,user_id):
    path=Path(path)
    if path.stat().st_size>10*1024*1024: raise ValueError('memory_file_too_large')
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or not isinstance(data.get(str(user_id)),dict):
        raise ValueError('selected_user_not_found')
    user=data[str(user_id)]; profile={}
    for key in ('name','nickname','last_seen'):
        if isinstance(user.get(key),str): profile[key]=user[key][:200]
    keywords=user.get('brain_keywords',{})
    profile['brain_keywords']={k[:80]:v for k,v in list(keywords.items())[:64]
        if isinstance(k,str) and type(v) in (int,float) and math.isfinite(v)} if isinstance(keywords,dict) else {}
    history=[]
    entries=user.get('history',[])
    if not isinstance(entries,list): raise ValueError('invalid_user_history')
    for item in entries[-40:]:
        if not isinstance(item,dict): continue
        text=item.get('content'); role=item.get('role')
        if role in ('user','assistant') and isinstance(text,str) and not text.lstrip().startswith('/'):
            history.append({'role':role,'content':text[:2048]})
    return {'profile':profile,'history':history[-12:]}
