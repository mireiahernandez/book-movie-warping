import torch as th
from torch import nn
from torch.nn.functional import grid_sample
import numpy as np


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, warped_feats):
        # norm = th.norm(feats - warped_feats, dim=0)
        # return norm.mean()
        return th.nn.functional.l1_loss(feats, warped_feats)

class CosineDistanceLoss(nn.Module):
    '''
    The original loss maximized the cosine similarity of the
    image and text embeddings of the N real pairs
    in the batch while minimizing the cosine similarity of the
    embeddings of the N2 − N incorrect pairings.
    A symmetric cross entropy loss over these similarity
    scores is optimized.
    Here, we take the ALIGNED pairs, using the ground truth book and movie alignments.
    '''
    def __init__(self):
        super().__init__()
        # from trained CLIP model
        self.temperature = 100

    def forward(self, feats, warped_feats):
        logits1 = self.temperature * feats.T @ warped_feats
        logits2 = self.temperature * warped_feats.T @ feats
        # symmetric loss function
        labels = th.LongTensor(np.arange(feats.shape[1]))
        loss_i = th.nn.functional.cross_entropy(logits1, labels)
        loss_t = th.nn.functional.cross_entropy(logits2, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    # # feats 512 x N
    # def forward(self, feats, warped_feats):
    #     cosine = 1 - th.diag(th.matmul(feats.T, warped_feats))
    #     return cosine.mean()

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
