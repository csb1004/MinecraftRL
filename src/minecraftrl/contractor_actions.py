"""Conservative non-GUI VPT contractor action adapter, not a PPO importer.

Camera units follow OpenAI's released converter (360/2400); keep floats.
Keep null/GUI rows in sequence for frame alignment and recurrent warm-up.
Eligibility is only action-format eligibility, not verified goal annotation.
"""
import math

KEYS=dict(zip(('escape','s','q','w','e','space','a','d','left.shift','left.control','f'),
              ('ESC','back','drop','forward','inventory','jump','left','right','sneak','sprint','swapHands')))
KEYS.update({str(i):f'hotbar.{i}' for i in range(1,10)})
KEYS={'key.keyboard.'+k:v for k,v in KEYS.items()}
BUTTONS=tuple(dict.fromkeys((*KEYS.values(),'attack','use','pickItem')))


class ContractorActions:
    def __init__(self):
        self.last_tick=None
        self.hotbar=None
        self.attack_stuck=False

    def convert(self,row):
        tick=row['tick'];hotbar=row['hotbar'];mouse=row['mouse'];keys=row['keyboard']['keys']
        if type(tick) is not int or tick<0 or (self.last_tick is not None and tick!=self.last_tick+1):
            raise ValueError('noncontiguous contractor tick; reset only at a declared segment boundary')
        if type(hotbar) is not int or not 0<=hotbar<=8:raise ValueError('invalid hotbar')
        if type(row['isGuiOpen']) is not bool:raise ValueError('invalid GUI flag')
        for k in ('dx','dy'):
            if type(mouse[k]) not in (int,float) or not math.isfinite(mouse[k]):raise ValueError('invalid mouse delta')
        if not isinstance(keys,list) or not all(isinstance(k,str) for k in keys):raise ValueError('invalid keys')
        for k in ('buttons','newButtons'):
            if not isinstance(mouse[k],list) or not all(type(v) is int and 0<=v<=2 for v in mouse[k]):
                raise ValueError('invalid mouse buttons')
        if self.last_tick is None:self.attack_stuck=mouse['newButtons']==[0]
        elif 0 in mouse['newButtons']:self.attack_stuck=False
        action={k:0 for k in BUTTONS}
        unknown=sorted(set(keys)-KEYS.keys())
        for key in keys:
            if key in KEYS:action[KEYS[key]]=1
        for index,key in enumerate(('attack','use','pickItem')):
            action[key]=int(index in mouse['buttons'] and not (index==0 and self.attack_stuck))
        # First hotbar is an observed state, not evidence of a selection action.
        if self.hotbar is not None and hotbar!=self.hotbar:action[f'hotbar.{hotbar+1}']=1
        action['camera']=[float(mouse['dy'])*.15,float(mouse['dx'])*.15]
        unsupported=[k for k in ('ESC','swapHands','pickItem') if action[k]]
        eligible=not row['isGuiOpen'] and not unknown and not unsupported
        self.last_tick=tick;self.hotbar=hotbar
        return dict(tick=tick,action=action,bc_eligible=eligible,unknown_keys=unknown,
                    unsupported_actions=unsupported,gui=row['isGuiOpen'],
                    initial_attack_suppression=self.attack_stuck,
                    null_action=not any(v for k,v in action.items() if k!='camera') and not any(action['camera']))
