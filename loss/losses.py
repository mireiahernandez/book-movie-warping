import torch as th
from torch import nn
from torch.nn.functional import grid_sample






class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, warped_feats):
        norm = th.norm(feats - warped_feats, dim=0)
        return norm.mean()

