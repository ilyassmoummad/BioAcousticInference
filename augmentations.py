import torch
from torch import nn
import torchvision
import numpy as np
import random

def random_freq_shift(_spec, Fshift):

    n_frames = _spec.shape[-1]
    n_bands = _spec.shape[-2]

    deltaf = int(np.random.uniform(low=0.0, high=Fshift))

    if deltaf == 0:
        return _spec

    _spec_out = torch.zeros_like(_spec).to(_spec.device)

    # new high band
    _spec_out[..., deltaf:,:] = _spec[..., :n_bands-deltaf,:]
    # new low band
    _spec_out[..., :deltaf,:] = _spec[..., -deltaf:,:]
    
    return _spec_out

class FreqShift(nn.Module):
    def __init__(self, Fshift=10):
        super().__init__()
        self.Fshift = Fshift

    def forward(self, spec):
        return random_freq_shift(spec, self.Fshift)

def mix_random(x):
    alpha = np.random.beta(5, 2, 1)[0]
    return alpha * x + (1. - alpha) * x[torch.randperm(x.shape[0]),...]

def compander(_spec, comp_alpha=0.75):

    if comp_alpha < 1:
        # compress amplitude
        adjust = np.random.uniform(low=comp_alpha, high=1.0)
    elif comp_alpha > 1:
        # expand amplitude
        adjust = np.random.uniform(low=1.0, high=comp_alpha)
    else:
        # bypass
        adjust = 1

    # apply compander
    _spec_out = _spec * adjust

    return _spec_out    

class Compander(torch.nn.Module):
    def __init__(self, do_compansion=True, comp_alpha=0.7):
        super().__init__()

        self.do_compansion = do_compansion
        self.comp_alpha = comp_alpha

    def forward(self, spec):
        if self.do_compansion:
            return compander(spec, self.comp_alpha)
        else:
            return spec

class RandomCrop(torch.nn.Module):
    def __init__(self, n_mels=128, time_steps=17, tcrop_ratio=0.6): #config for training set
        super().__init__()
        self.n_mels = n_mels
        self.tcrop_ratio = tcrop_ratio
        self.time_steps = time_steps

    def forward(self, audio):
        crop_ratio = random.uniform(self.tcrop_ratio, 1.0)
        new_time_steps = int(round(crop_ratio*self.time_steps))
        start = random.randint(0, self.time_steps-new_time_steps)   
        return audio[...,:self.n_mels,start:start+new_time_steps]

class Resize(torch.nn.Module):
    def __init__(self, n_mels=128, time_steps=17):
        super().__init__()
        self.n_mels = n_mels
        self.time_steps = time_steps
        
    def forward(self, mels):
        return torchvision.transforms.Resize((self.n_mels, self.time_steps), antialias=True)(mels)