"""Probability of the command delivered by the official hierarchical action mapper."""
import torch


class DeliveredActionLikelihood(torch.nn.Module):
    def __init__(self,mapper):
        super().__init__()
        groups={}
        for index,(buttons,off) in enumerate(zip(mapper.BUTTON_IDX_TO_FACTORED,
                                                mapper.BUTTON_IDX_TO_CAMERA_META_OFF)):
            group=groups.setdefault(tuple(buttons),{})
            if bool(off) in group:raise ValueError('duplicate mapper encoding in one camera gate')
            group[bool(off)]=index
        off_indices=[];on_indices=[];physical_groups=[]
        group_ids={buttons:index for index,buttons in enumerate(groups)}
        for buttons in mapper.BUTTON_IDX_TO_FACTORED:
            key=tuple(buttons);group=groups[key]
            off_indices.append(group.get(True,-1));on_indices.append(group.get(False,-1))
            physical_groups.append(group_ids[key])
        self.register_buffer('off_indices',torch.tensor(off_indices,dtype=torch.long))
        self.register_buffer('on_indices',torch.tensor(on_indices,dtype=torch.long))
        self.register_buffer('physical_groups',torch.tensor(physical_groups,dtype=torch.long))
        self.register_buffer('camera_off',torch.tensor(mapper.BUTTON_IDX_TO_CAMERA_META_OFF,dtype=torch.bool))
        self.camera_null=int(mapper.camera_null_idx)

    def effective_camera(self,actions):
        return torch.where(self.camera_off[actions['buttons']],self.camera_null,actions['camera'])

    def forward(self,pd,actions):
        buttons=actions['buttons'];camera=self.effective_camera(actions)
        def probability(indices):
            selected=indices[buttons]
            values=pd['buttons'].gather(-1,selected.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            return values.masked_fill(selected<0,float('-inf'))
        off=probability(self.off_indices);on=probability(self.on_indices)
        camera_logp=pd['camera'].gather(-1,camera.unsqueeze(-1)).squeeze(-1)
        # With camera off, every camera-head sample maps to the same command.
        # Camera on + the null camera sample also yields that exact command.
        result=torch.where(camera==self.camera_null,torch.logaddexp(off,on+camera_logp),on+camera_logp)
        return result.squeeze(-1)

    def matches(self,first,second):
        same_buttons=self.physical_groups[first['buttons']]==self.physical_groups[second['buttons']]
        return (same_buttons & (self.effective_camera(first)==self.effective_camera(second))).squeeze(-1)

    def mode(self,pd):
        """Most probable delivered command, after summing equivalent latent encodings."""
        buttons,camera=pd['buttons'],pd['camera']
        if (buttons.shape[:-1]!=camera.shape[:-1] or buttons.shape[-1]!=len(self.off_indices)
                or camera.shape[-1]<=self.camera_null):raise ValueError('invalid action distribution shape')
        def partners(indices):
            return buttons.index_select(-1,indices.clamp_min(0)).masked_fill(indices<0,float('-inf'))
        off,on=partners(self.off_indices),partners(self.on_indices)
        zero_camera=torch.logaddexp(off,on+camera[...,self.camera_null:self.camera_null+1])
        nonzero_camera=camera.clone();nonzero_camera[...,self.camera_null]=float('-inf')
        best_camera,camera_index=nonzero_camera.max(-1,keepdim=True)
        scores=torch.cat([zero_camera,on+best_camera],-1)
        if not torch.isfinite(scores.max(-1).values).all():raise ValueError('no finite delivered command')
        choice=scores.argmax(-1);count=buttons.shape[-1];rotating=choice>=count;group=choice%count
        off_index,on_index=self.off_indices[group],self.on_indices[group]
        chosen_button=torch.where(rotating,on_index,torch.where(off_index>=0,off_index,on_index))
        chosen_camera=torch.where(rotating,camera_index.squeeze(-1),torch.full_like(choice,self.camera_null))
        return dict(buttons=chosen_button,camera=chosen_camera)
