"""Bounded extractive dialogue summary; never an execution authority."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid

CATEGORIES=('preference','unconfirmed_proposal','question','task_context','context','assistant')


def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


@contextmanager
def file_lock(path):
    # Native process lock releases on process death. Use a single OS for writers
    # to one dialogue file; cross Windows/WSL locking is not assumed interoperable.
    with Path(path).open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0: stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


class DialogueMemory:
    def __init__(self,path=None):
        self.path=Path(path) if path else None
        self.data=self._read()

    def _read(self):
        if self.path is None or not self.path.exists():
            return {'schema':1,'revision':0,'recent':[],'summary':[]}
        if self.path.stat().st_size>131072: raise ValueError('dialogue_memory_too_large')
        value=json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(value,dict) or set(value)!={'schema','revision','recent','summary'} or value['schema']!=1:
            raise ValueError('invalid_dialogue_memory')
        if type(value['revision']) is not int or value['revision']<0: raise ValueError('invalid_memory_revision')
        for key,limit in [('recent',12),('summary',24)]:
            if not isinstance(value[key],list) or len(value[key])>limit: raise ValueError('invalid_memory_window')
            for row in value[key]:
                if not isinstance(row,dict) or row.get('category') not in CATEGORIES or row.get('role') not in ('user','assistant'):
                    raise ValueError('invalid_memory_entry')
                field='content' if key=='recent' else 'excerpt'
                if not isinstance(row.get(field),str) or len(row[field])>(2048 if key=='recent' else 240):
                    raise ValueError('invalid_memory_text')
                if not isinstance(row.get('source_id'),str): raise ValueError('missing_memory_source')
        return value

    @property
    def history(self): return [{'role':x['role'],'content':x['content']} for x in self.data['recent']]
    @property
    def summary(self): return copy.deepcopy(self.data['summary'])
    def export(self): return copy.deepcopy(self.data)

    def add(self,role,text,category='context'):
        if role not in ('user','assistant') or not isinstance(text,str) or category not in CATEGORIES:
            raise ValueError('invalid_dialogue_entry')
        if self.path:
            with file_lock(self.path.with_suffix('.lock')):
                if self._read()['revision']!=self.data['revision']: raise ValueError('stale_dialogue_writer_reopen_console')
                self._add(role,text,category)
        else: self._add(role,text,category)

    def _add(self,role,text,category):
        value=self.export(); entry={'source_id':uuid.uuid4().hex,'role':role,'content':text[:2048],
                                   'category':category,'authority':'dialogue_only'}
        value['recent'].append(entry)
        if len(value['recent'])>12:
            old=value['recent'].pop(0)
            note={k:v for k,v in old.items() if k!='content'}; note['excerpt']=old['content'][:240]
            note['truncated']=len(old['content'])>240
            value['summary'].append(note)
            matching=[i for i,x in enumerate(value['summary']) if x['category']==old['category']]
            if len(matching)>4: value['summary'].pop(matching[0])
        value['revision']+=1
        if self.path:
            # Source archive stays local. It is not all forwarded to the model.
            with self.path.with_suffix('.archive.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(entry,revision=value['revision']),ensure_ascii=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            atomic_json(self.path,value)
        self.data=value
