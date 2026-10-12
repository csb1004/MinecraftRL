"""Conservative, observable approach-stall candidate; never emits game actions."""
from collections import deque
import math


class ApproachRecovery:
    window=200
    span_limit=2.
    progress_floor=.5
    recovery_progress=1.

    def __init__(self):
        self.history=deque();self.active=False;self.target=None;self.entry_distance=None

    def update(self,obs,stage,*,terminal=False):
        values=[obs.get('server_'+k) for k in 'xyz']+[obs.get('wood_v3_target_'+k) for k in 'xyz']
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
            raise ValueError('recovery position/target telemetry unavailable')
        pos=tuple(values[:3]);target=tuple(values[3:]);distance=obs['wood_v3_distance'];tick=obs['server_tick']
        event=None;evidence=None
        if terminal:
            self.active=False;self.history.clear();event='terminal'
        elif self.active:
            if obs['wood_v3_kind']==1 and target!=self.target:event='target_changed'
            elif stage=='mine':event='mining_ready'
            elif stage=='collect':event='collection'
            elif obs['wood_v3_kind']==1 and distance<=self.entry_distance-self.recovery_progress:event='approach_progress'
            if event:self.active=False;self.history.clear()
        if not terminal and not self.active:
            if stage!='approach':self.history.clear()
            else:
                if self.history and target!=self.history[-1][3]:self.history.clear()
                self.history.append((tick,pos,distance,target))
                while self.history and self.history[0][0]<tick-self.window:self.history.popleft()
                if self.history and tick-self.history[0][0]>=self.window:
                    spans=[max(row[1][axis] for row in self.history)-min(row[1][axis] for row in self.history) for axis in range(3)]
                    progress=self.history[0][2]-min(row[2] for row in self.history)
                    if max(spans)<=self.span_limit and progress<self.progress_floor:
                        self.active=True;self.target=target;self.entry_distance=distance;event='entered'
                        evidence=dict(span=spans,best_distance_progress=progress,raw_distance=distance,target=target)
                        self.history.clear()
        return dict(active=self.active,event=event,evidence=evidence,
                    window_ticks=0 if not self.history else tick-self.history[0][0])
