from scipy.stats import norm
import numpy as np
import torch as th
import ipdb

def kernel(t, s):
    # gaussian centered at t with mean 0 and std 1
    # evaluated at s
    ipdb.set_trace()
    return norm.pdf(s, loc=t, scale=1)


############ Forward warping function ################
'''
returns warped src image according to f_times
    src is a tensor of shape 512 x N_book
    dst is a tensor of 512 x N_movie
    f_times is a tensor of length N_book
'''
def forward_warping(src, dst, f_times):
    len_src, len_dst = src.shape[1], dst.shape[1]
    ksum = th.zeros(size=(len_dst,))
    # Loop through times in src image
    t_src = 0
    while t_src < len_src:
        t_dst = f_times[t_src]
        # Loop through times in dst image
        s_dst = 0
        while s_dst < len_dst:
            dst[:, s_dst] += kernel(t_dst, s_dst)*src[:,t_src]
            ksum[s_dst] += kernel(t_dst, s_dst)
            s_dst =+ 1
        t_src += 1
    
    # Normalize
    s_dst = 0
    while s_dst < len_dst:
        dst[:, t_dst] /= ksum[s_dst]
        s_dst += 1
    
    return dst

