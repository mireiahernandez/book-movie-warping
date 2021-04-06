import torch as th
from torch.nn.functional import grid_sample
import ipdb
import numpy as np


# input features: shape (512, len_input)
# output_times: shape (len_output, ): for each output time, time in the input feature in the range[0,1]
def inverse_warping(input_features, output_times):
    len_feats = input_features.shape[0]
    list_warped_input_features = []
    for i in range(len_feats):
        warped_input_features_1_dim = inverse_warping_1_dim(input_features[i].unsqueeze(0), output_times)
        list_warped_input_features.append(warped_input_features_1_dim) 
    warped_input_features = th.cat(list_warped_input_features, dim=0)
    return warped_input_features


# input features: shape (1, len_input)
# output_times: shape (len_output, ): for each output time, time in the input feature in the range[0,1]
def inverse_warping_1_dim(input_features, output_times):
    # get output dimension
    len_output = output_times.shape[0]

    # scale outputs so they range between -1 and 1
    output_times_scaled = 2 * output_times - 1
    
    # get single dimension times (uniformly distributed between -1 and 1)
    single_dim_times = np.array(list(range(0, len_output))) / (len_output-1)
    single_dim_times = th.FloatTensor(single_dim_times)
    single_dim_times = 2 * single_dim_times - 1
    
    # Calculate grid of shape (N, Hout, Wout, 2) = (1, 1, len_output, 2)
    grid = th.zeros(size=(1, 1, len_output, 2))
    
    grid[:,:,:,0] = single_dim_times
    grid[:,:, :, 1] = output_times_scaled

    # Reshape input features to dimension (N, C, Hin, Win) = (1,1,512,len_input)
    input_features = input_features.unsqueeze(0).unsqueeze(0)

    # Do inverse image warping
    output_features = grid_sample(input_features, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=True)

    # Resize output features to (512, len_output)
    output_features =output_features.squeeze(0).squeeze(0)

    return output_features

def linear_kernel(dif):
    return th.max(th.ones(size=dif.shape) - dif, th.zeros(size=dif.shape))

def gaussian_kernel(dif):
    # gaussian centered at t with mean 0 and std 1
    # evaluated at s
    std = 0.5
    norm = std*th.sqrt(th.FloatTensor([2*np.pi]))
    return th.exp(-th.square(dif/std)/2) / norm

def reverse_mapping(src, invf_times, kernel_type):
    # Get dst and src lengths (1-dim)
    len_dst = invf_times.shape[0]
    len_src = src.shape[1]
    len_feat = src.shape[0]
    # Initialize a tensor of zeros for dst image
    dst = th.zeros(size=(src.shape[0], len_dst))
    
    # Obtain a grid of pairs of integer pixels and inverse mapped pixels
    src_times = th.FloatTensor(np.arange(len_src))
    pairs = th.cartesian_prod(src_times, invf_times).reshape((len_src, len_dst, 2))

    # obtain the absolute value of the difference between elements of the said pairs
    abs_dif = th.abs(pairs[:,:,0] - pairs[:,:,1])
    
    # calculate the weights matrix
    # weights[i,j]: weight that multiplies src[i] in dst[j]
    # shape (len_src, len_dst)
    if kernel_type == "gaussian": weights = gaussian_kernel(abs_dif)
    elif kernel_type == "linear": weights = linear_kernel(abs_dif)

    # calculate
    # iterate pixels x over destination image dst
    '''
    for x in range(len_dst):
        for u in range(len_src):
            dst[:, x] += th.mul(src[:, u], weights[u, x])
    '''
     # Flatten to obtain (--- len_src----)(----len_src----)....
    weights = weights.T.flatten()

    # Do a convolution of kernel_vector and src image to obtain dst
    conv1d = th.nn.Conv1d(in_channels=1, out_channels=len_feat, kernel_size=len_src,
                    stride=len_src, padding=0, dilation=1, groups=1, bias=False,
                    padding_mode='zeros')
    
    conv1d.weight = th.nn.Parameter(src.unsqueeze(1))
    dst = conv1d(weights.unsqueeze(0).unsqueeze(0)).squeeze(0)
    
    return dst

