"""Explicit scripted cost calibration, never a learned-policy replacement."""
import math
from .runtime import run_episode,write_json,jsonable
from .rl_collect import RLEnvironment
from .rl_core import LOGS


def aim(gui,x,y,z):
    dx=x-gui['eye_x'];dy=y-gui['eye_y'];dz=z-gui['eye_z']
    yaw=math.degrees(math.atan2(-dx,dz));pitch=-math.degrees(math.atan2(dy,math.hypot(dx,dz)))
    turn=(yaw-gui['yaw']+180)%360-180;tilt=pitch-gui['pitch']
    return [max(-10,min(10,tilt)),max(-10,min(10,turn))],max(abs(turn),abs(tilt))<2


def navigate(gui,x,z,jump=False):
    dx=x-gui['server_x'];dz=z-gui['server_z']
    if math.hypot(dx,dz)<.22:return {},True
    yaw=math.degrees(math.atan2(-dx,dz));turn=(yaw-gui['yaw']+180)%360-180
    return dict(camera=[max(-10,min(10,-gui['pitch'])),max(-10,min(10,turn))],
                forward=int(abs(turn)<12),jump=int(jump)),False


class ProbePolicy:
    def __init__(self,raw,world,route):
        if world.get('kind') not in ('break','detour','death','recovery') or route not in ('break','detour','death'):
            raise ValueError('probe requires an explicitly constructed arena')
        if (world['kind']=='death')!=(route=='death'):raise ValueError('death/route mismatch')
        if world['kind']=='recovery' and route!='detour':raise ValueError('bedrock calibration requires detour')
        if world.get('curriculum'):raise ValueError('legacy probe coordinates do not support curriculum arenas')
        self.raw=raw;self.world=world;self.route=route;self.rows=[];self.environment=None;self.milestones=[]
        self.ox,self.oz=[v*16 for v in world['chunk']]
    def reset(self,goal=None):
        gap=14.5 if self.world['kind'] in ('break','recovery') else 2.0
        self.waypoints=([(5.5,3.5),(5.5,gap),(9.5,gap),(9.5,3.5)] if self.route=='detour' else [(5.5,3.5)])
        self.dig_ticks=0;self.phase='move';self.milestones=[];self.harvest_target_seen=False;self.stump_seen=False
    def set_reward_context(self,context):self.context=list(context)
    def close(self):pass
    def commit(self,action,reward,terminal):
        self.rows.append(dict(action=jsonable(action),reward=reward,terminal=terminal,phase=self.phase))
    def act(self,screen):
        action=self.raw.action_space.noop();g=self.environment.gui
        if self.route=='death':return action
        if self.waypoints:
            x,z=self.waypoints[0];change,arrived=navigate(g,self.ox+x,self.oz+z,self.phase=='cross')
            action.update(change)
            if arrived:self.waypoints.pop(0)
            return action
        if self.phase=='move' and self.route=='break':self.phase='dig'
        elif self.phase in ('move','cross'):self.phase='harvest'
        if self.phase=='dig':
            camera,aligned=aim(g,self.ox+7.5,202.5,self.oz+3.5);action['camera']=camera
            if aligned:
                action['attack']=1;self.dig_ticks+=1
                limit=20 if self.world['kind']=='break' else 160
                if self.dig_ticks>=limit:
                    self.phase='cross';self.waypoints=[(9.5,3.5)]
                    self.milestones.append(dict(event='dig_schedule_finished',step=len(self.rows)+1))
            return action
        target_hit=(g['hit_x'],g['hit_y'],g['hit_z'])==(self.ox+11,202,self.oz+3)
        if self.phase=='harvest' and self.harvest_target_seen and not target_hit:self.phase='clear_stump'
        if self.phase=='clear_stump':
            stump_hit=(g['hit_x'],g['hit_y'],g['hit_z'])==(self.ox+11,201,self.oz+3)
            if self.stump_seen and not stump_hit:self.phase='collect'
            else:
                camera,aligned=aim(g,self.ox+11.5,201.5,self.oz+3.5);action['camera']=camera
                if aligned and stump_hit:self.stump_seen=True;action['attack']=1
                return action
        if self.phase=='collect':
            change,_=navigate(g,self.ox+11.5,self.oz+3.5);action.update(change)
            return action
        camera,aligned=aim(g,self.ox+11.5,202.5,self.oz+3.5);action['camera']=camera
        if aligned and target_hit:self.harvest_target_seen=True
        if aligned:action['attack']=1;action['forward']=int(g['server_x']<self.ox+10.65)
        return action


def run_probe(policy,env,output,*,world,gate,stop_file,**kwargs):
    wrapper=RLEnvironment(env,policy,strict_ticks=True,reward_mode='wood-potential-v2',
        is_stopped=lambda:gate.requested.is_set() or bool(stop_file and stop_file.exists()))
    policy.environment=wrapper
    report=run_episode(policy,wrapper,output,goal='fixture route calibration',target_items=LOGS,
        completion_check=lambda obs:wrapper.reward.completed,gate=gate,stop_file=stop_file,**kwargs)
    report.update(controller='scripted_fixture_probe',route=policy.route,learned_policy_loaded=False,
                  rl_training_eligible=False,rl_updates=0,initial_world=world,
                  server_target_reached=bool(wrapper.reward and wrapper.reward.completed),milestones=policy.milestones)
    write_json(output/'probe-rows.json',policy.rows)
    write_json(output/'initial-state.json',getattr(wrapper,'initial',{}))
    write_json(output/'last-observed-state.json',dict(
        observation={k:v for k,v in (wrapper.last or {}).items() if k!='pov'},
        server=wrapper.gui,may_include_rejected_transition=not report['execution_valid']))
    write_json(output/'summary.json',report)
    return report
