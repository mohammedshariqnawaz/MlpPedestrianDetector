import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NECKS
import numpy as np
from ..utils import window_partition, MixerBlock

import sys

@NECKS.register_module
class MLPFPN(nn.Module):
    """MLPFPN

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_dim=8,
                 start_index=1,
                 mixer_count=1,
                 start_stage=0,
                 end_stage=4):
        super(MLPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_index = start_index
        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count
        self.patch_dim = patch_dim
        self.start_stage = start_stage
        self.end_stage = end_stage

        pc = int(np.sum([self.in_channels[i] * 2**(2*(self.num_ins-1 - i)) for i in range(self.num_ins)[self.start_stage:self.end_stage]]))
        self.intpr = nn.Linear(pc, (self.patch_dim**2)*self.out_channels)

        self.mixers = None
        if self.mixer_count > 0:
            self.mixers = nn.Sequential(*[
                MixerBlock(self.patch_dim**2, self.out_channels) for i in range(self.mixer_count)
            ])

    def init_weights(self):
        pass

    def forward(self, inputs):
        assert len(inputs) == self.num_ins

        B, H4, W4, _ = inputs[0].shape
        parts = []
        # print("NUM_ins",self.num_ins)

        # print("INMPUTS", inputs[0].shape)

        for i in range(self.num_ins)[self.start_stage:self.end_stage]:
            # print("i",inputs[i].shape)
            part = window_partition(inputs[i], 2**(self.num_ins-1 - i), channel_last=False)
            # print("i++",part.shape)
            # print("Asdasdss",part.shape)
            parts.append(torch.flatten(part, -2))
            
            
        
        # print("OUT",out)
        # for i in range(len(parts)):
            # print("PARTS",parts[i].shape)
        
        out = torch.cat(parts, dim=-1)
        # print("OUT",out.shape)
        
        out = self.intpr(out)
        # print("OUT_AFTER_INTERPOLATIOn",out.shape)
        B, T, _ = out.shape
        outputs = out.view(B, T, self.patch_dim**2, self.out_channels)

        # print("OUTPUT",outputs.shape)
        

        if self.mixers is not None:
            outputs = self.mixers(outputs)
        # print("LAST",tuple([outputs.shape]))
        
        return tuple([outputs])
