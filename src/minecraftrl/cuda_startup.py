"""Prepare the convolution backend without invoking policy memory or sampling."""
import time


def prime_convolution(layer):
    import torch
    if not isinstance(layer,torch.nn.Conv2d):
        raise TypeError('a convolution layer is required')
    cuda=layer.weight.is_cuda
    started=time.monotonic()
    # Match the guided policy's NHWC-to-NCHW view without drawing random values.
    image=torch.zeros((2,128,128,layer.in_channels),device=layer.weight.device).permute(0,3,1,2)
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=cuda):
        output=layer(image)
    if cuda:torch.cuda.synchronize(layer.weight.device)
    if not torch.isfinite(output).all():raise ValueError('nonfinite convolution preparation')
    return dict(completed=True,seconds=time.monotonic()-started,device=str(layer.weight.device),
        input_shape=list(image.shape),input_stride=list(image.stride()),
        scope='convolution backend only; no recurrent state, sampling, or environment action')
