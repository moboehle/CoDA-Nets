import torch
from torch import nn


class Squashing(nn.Module):

    def __init__(self):
        """
        This module calculates the norm applied by the squashing function
            (see Dynamic Routing Between Capsules by Sabour et al.)
        along the first dimension of the input.
        """
        super().__init__()

    def forward(self, in_tensor):

        norm = torch.norm(in_tensor, 2, dim=1, keepdim=True)
        norm = (1 + norm.pow(2)) / norm

        return norm


class LpNorm(nn.Module):

    def __init__(self, p=1, return_norm=False, **kwargs):
        _ = kwargs
        super().__init__()
        self.p = p
        self.return_norm = return_norm

    def forward(self, in_tensor):
        if self.return_norm:
            return torch.norm(in_tensor, p=self.p, dim=1, keepdim=True)
        out = (in_tensor / torch.norm(in_tensor, p=self.p, dim=1, keepdim=True))
        return out

    def __str__(self):
        return "LpNorm(p={p})".format(p=self.p)

    def __repr__(self):
        return self.__str__()


class L2Norm(LpNorm):

    def __init__(self, **kwargs):
        super().__init__(p=2, **kwargs)
