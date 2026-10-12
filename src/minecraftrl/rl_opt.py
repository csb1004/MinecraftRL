"""Bounded PPO update of STEVE-1's action head on frozen recurrent features."""
def ppo_loss(logp,old_logp,advantage,clip):
    import torch
    ratio=(logp-old_logp).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
