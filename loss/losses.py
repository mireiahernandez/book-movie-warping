import torch as th
from torch import nn
from torch.nn.functional import grid_sample






class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, warped_feats):
#        norm = th.norm(feats - warped_feats, dim=0)
#        return norm.mean()
        return th.nn.functional.l1_loss(feats, warped_feats)
#        return th.nn.functional.mse_loss(feats, warped_feats)

class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    # feats 512 x N
    def forward(self, feats, warped_feats):
        cosine = 1 - th.diag(th.matmul(feats.T, warped_feats))
        return cosine.mean()

class GTDifLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_times, gt_times):
        dif = th.abs(output_times - gt_times)
        return dif.mean()

class GTNormLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_times, gt_times):
        norm = th.norm(output_times - gt_times, dim=0) / output_times.shape[0]
        return norm
