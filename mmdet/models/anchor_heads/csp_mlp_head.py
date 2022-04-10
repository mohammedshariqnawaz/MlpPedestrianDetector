import torch
import torch.nn as nn

from ..registry import HEADS
from ..utils import Scale, window_reverse, window_partition, MixerBlock
from .csp_head import CSPHead
import numpy as np

INF = 1e8


@HEADS.register_module
class CSPMLPHead(CSPHead):

    def __init__(self, *args, patch_dim=4, windowed_input=True, **kwargs):
        self.patch_dim = patch_dim
        super(CSPMLPHead, self).__init__(*args, **kwargs)
        self.windowed_input = windowed_input

    def _init_layers(self):
        self.mlp_with_feat_reduced = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.in_channels),
            nn.Linear(self.in_channels, self.feat_channels)
        )

        self.pos_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 1),
        )

        self.reg_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 1)
        )

        self.off_mlp = nn.Sequential(
            MixerBlock(self.patch_dim**2, self.feat_channels),
            nn.Linear(self.feat_channels, 2)
        )

    def init_weights(self):
        self.reg_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.offset_scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError(
            f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_single(self, x, reg_scale, offset_scale):
        # print("x into head",x.shape)
        if not self.windowed_input:
            windows = window_partition(x, self.patch_dim, channel_last=False)
            
        else:
            windows = x
        print("windows",windows.shape)

        feat = self.mlp_with_feat_reduced(windows)
        print("feat",feat.shape)
        x_cls = self.pos_mlp(feat)
        x_reg = self.reg_mlp(feat)
        x_off = self.off_mlp(feat)

        print("x_cls",x_cls.shape)
        print("x_reg",x_reg.shape)
        print("x_off",x_off.shape)
        h = int(2**((np.log2(feat.shape[1])-1)/2)) * self.patch_dim
        w = int(h*2)

        x_cls = window_reverse(x_cls, self.patch_dim, h, w)
        x_reg = window_reverse(x_reg, self.patch_dim, h, w)
        x_off = window_reverse(x_off, self.patch_dim, h, w)

        print("x_cls after window operation",x_cls.shape)
        print("x_reg after window operation",x_reg.shape)
        print("x_off after window operation",x_off.shape)

        return x_cls, reg_scale(x_reg).float(), offset_scale(x_off).float()
