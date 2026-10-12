"""Privileged scripted demonstrations, never learned-policy evaluation or PPO data."""
from copy import deepcopy
from types import SimpleNamespace

from .fixture_probe import ProbePolicy
from .rl_collect import RLEnvironment
from .rl_core import LOGS
from .runtime import run_episode, write_json, jsonable


def validate_teacher_world(world,split):
    role=world.get('experimental_split')
    allowed=(role=='train' and split=='train' and world.get('seed') in (104743,104759)) or (
        role=='development' and split=='eval' and world.get('seed')==104761)
    if not allowed or world.get('kind')!='recovery':
        raise ValueError('teacher requires a reserved train/development recovery arena')
    curriculum=world.get('curriculum',{})
    if type(curriculum.get('wall_end')) is not int or not 1<=curriculum['wall_end']<=12:
        raise ValueError('unsupported teacher geometry')
    if type(curriculum.get('mirror_z')) is not bool:raise ValueError('missing mirror geometry')


class CorrectionTeacher(ProbePolicy):
    def __init__(self,raw,world):
        validate_teacher_world(world,'eval' if world.get('experimental_split')=='development' else 'train')
        canonical=dict(world);canonical.pop('curriculum',None)
        super().__init__(raw,canonical,'detour')
        self.original_world=world
        self.mirror=world['curriculum']['mirror_z']

    def reset(self,goal=None):
        super().reset(goal)
        gap=self.original_world['curriculum']['wall_end']+1.5
        self.waypoints=[(5.5,3.5),(5.5,gap),(9.5,gap),(9.5,3.5)]

    def act(self,screen):
        actual=self.environment
        self.before=deepcopy(actual.gui)
        self.before_context=list(self.context)
        canonical=dict(actual.gui)
        if self.mirror:
            for key in ('server_z','eye_z'):canonical[key]=2*self.oz+16-canonical[key]
            canonical['hit_z']=2*self.oz+15-canonical['hit_z']
            canonical['yaw']=180-canonical['yaw']
        self.environment=SimpleNamespace(gui=canonical)
        try:action=super().act(screen)
        finally:self.environment=actual
        if self.mirror and 'camera' in action:
            action['camera']=[action['camera'][0],-action['camera'][1]]
        return action

    def commit(self,action,reward,terminal):
        super().commit(action,reward,terminal)
        self.rows[-1].update(server_before=jsonable(self.before),server_after=jsonable(self.environment.gui),
                             reward_context=self.before_context,frame=f"frames/{len(self.rows)-1:06d}.png")


def run_teacher(policy,env,output,*,world,gate,stop_file,**kwargs):
    policy.output=output
    wrapper=RLEnvironment(env,policy,strict_ticks=True,reward_mode='wood-potential-v4',
        is_stopped=lambda:gate.requested.is_set() or bool(stop_file and stop_file.exists()))
    policy.environment=wrapper
    report=run_episode(policy,wrapper,output,goal='collect wood',target_items=LOGS,
        completion_check=lambda obs:wrapper.reward.completed,gate=gate,stop_file=stop_file,**kwargs)
    release=getattr(env,'release_evidence',{})
    eligible=bool(report['execution_valid'] and report['status']=='target_reached' and
                  wrapper.reward.completed and release.get('verified') and release.get('held_key_count')==0)
    report.update(controller='privileged_scripted_correction_teacher',learned_policy_loaded=False,
                  rl_training_eligible=False,rl_updates=0,initial_world=world,
                  server_target_reached=bool(wrapper.reward and wrapper.reward.completed),
                  demonstration_candidate=eligible,bc_training_ready=False,
                  teacher_observations='server position, ray hit, known fixture geometry; not video-only',
                  server_tick_deltas=wrapper.ticks)
    write_json(output/'teacher-rows.json',policy.rows)
    write_json(output/'initial-state.json',getattr(wrapper,'initial',{}))
    write_json(output/'last-observed-state.json',dict(server=wrapper.gui))
    write_json(output/'summary.json',report)
    return report
