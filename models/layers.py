import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#from timm.models.layers import trunc_normal_, DropPath

from copy import deepcopy


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
"""

class CNBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

class GRN(nn.Module):

    
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1,1,1, dim))
        self.beta = nn.Parameter(torch.zeros(1,1,1,dim))
        
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x




"""