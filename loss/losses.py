import torch as th
from torch import nn
from torch.nn.functional import grid_sample
import numpy as np
import ipdb

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, warped_feats):
        return th.nn.functional.l1_loss(feats, warped_feats)

class SimilarityDialog(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, similarity_dialogs, f_theta_dialogs, dialog_source_index, temperature=1, device='cpu:0'):
        '''
        Compute the similarity for the predicted dialogs weighted by their distance
        similarity_dialogs - dot product between normalized dialog features
        f_theta_dialogs - f_theta evaluated at dialogs in the source domain
        dialog_source_index - index of the source dialog
        temperature - parameter in weighting kernel
 
        '''
        src_times = dialog_source_index
        invf_times = f_theta_dialogs
        len_src = len(src_times)
        len_dst = len(invf_times)
        pairs = th.cartesian_prod(src_times, invf_times.to(device)).reshape((len_src, len_dst, 2))
#        ipdb.set_trace()
    # obtain the absolute value of the difference between elements of the said pairs
        abs_dif = th.abs(pairs[:, :, 0] - pairs[:, :, 1])
        normalized = 1- (abs_dif-th.min(abs_dif, 0)[0])/(th.max(abs_dif, 0)[0]-th.min(abs_dif, 0)[0])
        weight = th.nn.functional.softmax(temperature * normalized, 0)
        sim = th.mul(similarity_dialogs, weight)#.sum(0).mean()
        sim = th.max(sim, 0)[0].mean()
        return -sim, th.min(abs_dif, 0)[0].mean()



class CosineDistanceLoss(nn.Module):
    '''
    Maximizing the cosine similarity of the warped features
    '''
    def __init__(self, device='cpu:0'):
        super().__init__()
        self.device = device

    def forward(self, feats, warped_feats):
        return -th.mul(feats, warped_feats).sum(0).mean()

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
