"""One-request Windows/WSL mailbox. No key, network, or action dispatch here."""
from copy import deepcopy
import json
from pathlib import Path

from .memory import atomic_json
from .recovery_draft import validate_packet


def read_bounded(path,limit=32768):
    with Path(path).open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('recovery exchange message too large')
    return json.loads(raw.decode('utf-8'))


class FileRecoveryProvider:
    kind='windows_mailbox'
    choice='external_advisory_only'
    delay_s=0.
    def __init__(self,path,*,live=False):
        self.path=Path(path);self.calls=0;self.closed=False;self.packet=None;self.live=live
    @property
    def actual_llm_calls(self):
        # Only the Windows transport receipt can establish whether a live call was sent.
        return None if self.live and self.calls else 0
    def start(self,packet):
        if self.calls or self.closed:raise RuntimeError('recovery exchange already used or closed')
        validate_packet(packet)
        self.path.mkdir(exist_ok=False)
        self.packet=deepcopy(packet);self.calls=1
        atomic_json(self.path/'request.json',dict(schema=1,packet=self.packet))
    def poll(self):
        if self.closed or self.packet is None:return None
        path=self.path/'response.json'
        if not path.exists():return None
        reply=read_bounded(path)
        if (not isinstance(reply,dict) or set(reply)!={'request_id','proposal'}
                or reply['request_id']!=self.packet['request_id']):
            raise ValueError('recovery exchange identity/schema mismatch')
        return deepcopy(reply['proposal'])
    def close(self):
        if self.closed:return
        self.closed=True
        if self.packet is not None:
            atomic_json(self.path/'closed.json',dict(request_id=self.packet['request_id'],closed=True))


class FileRMPlanProvider:
    """Separate one-use mailbox per bounded plan request; no network in the game worker."""
    kind='windows_rm_plan_mailbox'
    choice='external_subgoal_plan'
    delay_s=0.
    def __init__(self,path,*,live=False,max_calls=1):
        if type(max_calls) is not int or not 1<=max_calls<=3:raise ValueError('invalid plan call budget')
        self.path=Path(path);self.live=live;self.maximum=max_calls
        self.calls=0;self.closed=False;self.packet=None;self.current=None;self.seen_requests=set()
    @property
    def actual_llm_calls(self):return None if self.live and self.calls else 0
    def start(self,packet):
        from .rm_planner import validate_packet
        if self.closed or self.calls>=self.maximum:raise RuntimeError('RM mailbox budget exhausted')
        validate_packet(packet)
        if packet.get('mode')=='archived_review':raise ValueError('archived analysis cannot enter live mailbox')
        if packet['request_id'] in self.seen_requests:raise RuntimeError('RM request already submitted')
        self.seen_requests.add(packet['request_id'])
        if not self.calls:self.path.mkdir(exist_ok=False)
        if self.current is not None:
            atomic_json(self.current/'closed.json',dict(request_id=self.packet['request_id'],closed=True))
        self.calls+=1;self.current=self.path/f'request-{self.calls:03d}'
        self.current.mkdir(exist_ok=False);self.packet=deepcopy(packet)
        atomic_json(self.current/'request.json',dict(schema=1,packet=self.packet))
    def poll(self):
        if self.closed or self.packet is None:return None
        path=self.current/'response.json'
        if not path.exists():return None
        value=read_bounded(path,65536)
        if (not isinstance(value,dict) or set(value)!={'request_id','proposal'}
                or value['request_id']!=self.packet['request_id']):
            raise ValueError('RM mailbox response mismatch')
        from .rm_planner import validate_plan
        return validate_plan(value['proposal'],self.packet)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.current is not None:
            atomic_json(self.current/'closed.json',dict(request_id=self.packet['request_id'],closed=True))
            atomic_json(self.path/'closed.json',dict(closed=True,requests=self.calls))
