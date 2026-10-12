"""Episode-preserving truncated BC with fresh current-weight burn-in."""
from contextlib import nullcontext
from .bc_core import validate_plan


def episode_rows(data):
    import torch
    validate_plan(data['plan'])
    correction=data['plan'].get('data_source')=='student_correction'
    if correction and data.get('bc_schema')!=2:raise ValueError('masked sequence schema required')
    if data['plan'].get('data_source') not in ('local_teacher','student_correction'):raise ValueError('complete demonstration episodes required')
    sessions=data['plan']['sessions']
    seeds={split:{s['seed'] for s in sessions if s['split']==split} for split in ('train','development')}
    if seeds['train'] & seeds['development']:raise ValueError('world seed crosses splits')
    groups={s['id']:[] for s in sessions}
    for row in data['rows']:
        if row['session'] not in groups:raise ValueError('unknown source episode')
        groups[row['session']].append(row)
    for session in sessions:
        rows=groups[session['id']]
        if len(rows)!=session['end']:raise ValueError('incomplete temporal episode')
        for i,row in enumerate(rows):
            if row['frame']!=i or row['split']!=session['split']:raise ValueError('temporal order/split mismatch')
            if correction:
                from .correction_bc import correction_owner
                eligible=session['frames'][i]['loss_eligible']
                if type(row.get('loss_eligible')) is not bool or row['loss_eligible']!=eligible or row.get('owner')!=correction_owner(session,i):
                    raise ValueError('cached correction mask/owner mismatch')
                if (row.get('action') is not None)!=eligible:raise ValueError('context frames cannot carry action targets')
            feature=row['features'];x=row['temporal_input']
            if feature.ndim!=3 or feature.shape[:2]!=(2,1) or x.shape!=feature[...,:-6].shape:
                raise ValueError('invalid CFG temporal feature shape')
            if not torch.isfinite(feature).all() or not torch.isfinite(x).all():raise ValueError('nonfinite temporal feature')
    return groups


def windows(length,size=8):
    if type(length) is not int or length<1 or type(size) is not int or size<1:raise ValueError('invalid sequence bounds')
    return [(start,min(start+size,length)) for start in range(0,length,size)]


def device_tree(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:device_tree(v,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(device_tree(v,device) for v in value)
    return value


def recurrent_step(block,row,memory,index):
    import torch
    device=next(block.parameters()).device
    with torch.cuda.amp.autocast() if device.type=='cuda' else nullcontext():
        return block(row['temporal_input'].to(device),
                     torch.full((2,1),index==0,dtype=torch.bool,device=device),memory)


def latent_features(tail,latent,row):
    import torch
    with torch.cuda.amp.autocast() if latent.device.type=='cuda' else nullcontext():
        result=tail(torch.relu(latent))
    return torch.cat([result.float(),row['features'][...,-6:].to(result.device)],-1)


def forward_window(block,tail,rows,initial,start,end):
    """No saved trained memory: recompute prefix, then retain gradients in the window."""
    import torch
    if not 0<=start<end<=len(rows):raise ValueError('invalid episode window')
    memory=device_tree(initial,next(block.parameters()).device)
    with torch.no_grad():
        for i in range(start):_,memory=recurrent_step(block,rows[i],memory,i)
    outputs=[]
    for i in range(start,end):
        latent,memory=recurrent_step(block,rows[i],memory,i)
        outputs.append(latent_features(tail,latent,rows[i]))
    return torch.cat(outputs,1)
