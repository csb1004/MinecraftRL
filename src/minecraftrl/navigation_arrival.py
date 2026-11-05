"""Optional consecutive settled-arrival evidence for the common movement skill."""
import math
from .rl_core import tick_delta
from .waypoints import arrived
from .navigation_observation import position,client_tick_delta


class ArrivalWindow:
    def __init__(self, required_ticks=1, *, pose_source='server'):
        if type(required_ticks) is not int or not 1 <= required_ticks <= 20:
            raise ValueError('arrival hold must be 1..20 committed server ticks')
        self.required_ticks = required_ticks
        if pose_source not in ('server','client'):raise ValueError('unknown arrival observation source')
        self.pose_source=pose_source
        self.count = 0

    def reset(self):
        self.count = 0

    def advance(self, before, after, target):
        tick_delta(before, after)
        if self.pose_source=='client':client_tick_delta(before,after)
        inside = arrived(after, target)
        if self.required_ticks == 1:
            self.count = int(inside)
            return inside
        # These are observed displacements, not predicted velocity or action intent.
        horizontal = math.hypot(after['server_x'] - before['server_x'],
                                after['server_z'] - before['server_z'])
        vertical = abs(after['server_y'] - before['server_y'])
        settled = (inside and horizontal <= .03 and vertical <= .01 and
                   abs(after['server_y'] - target[1]) <= .05)
        if self.pose_source=='client':
            first,last=position(before,'client'),position(after,'client')
            settled=settled and (
                math.hypot(last[0]-target[0],last[2]-target[2])<=.35 and
                math.hypot(last[0]-first[0],last[2]-first[2])<=.03 and
                abs(last[1]-first[1])<=.01 and abs(last[1]-target[1])<=.05 and
                after.get('client_on_ground')==1 and after.get('server_on_ground')==1)
        self.count = self.count + 1 if settled else 0
        return self.count >= self.required_ticks
