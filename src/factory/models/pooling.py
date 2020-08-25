import torch
import torch.nn as nn
import torch.nn.functional as F


# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        if self.dim == 2:
            return gem_2d(x, p=self.p, eps=self.eps)
        elif self.dim == 3:
            return gem_3d(x, p=self.p, eps=self.eps)
        elif self.dim == 1:
            return gem_1d(x, p=self.p, eps=self.eps)


class AdaptiveConcatPool2d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)


class AdaptiveConcatPool3d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool3d(x, 1), F.adaptive_max_pool3d(x, 1)), dim=1)