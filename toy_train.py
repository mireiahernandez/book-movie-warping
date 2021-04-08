import numpy as np
import torch as th
import ipdb
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import time
import torchvision
import io

import PIL.Image
from torchvision.transforms import ToTensor

from network.mlp import MLP
from warping.inverse_warping import reverse_mapping
from torch.nn.functional import grid_sample
from loss.losses import ReconstructionLoss, CosineDistanceLoss
from loss.losses import GTDifLoss, GTNormLoss

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal


class TimesDataset(Dataset):
    def __init__(self, times):
        self.times = times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.times[idx], idx



def get_plot(input_times, output_times, gt_times):
    plt.close()
    plt.plot(input_times, output_times, 'r-')
    plt.plot(input_times, gt_times, 'g-')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf


def plot_diff(text_feats, pred_output_ft, gt_output_feats):
    diff = pred_output_ft - gt_output_feats
    _min = min(text_feats.min(), toy_input_feats.min(), gt_output_feats.min(),
               diff.min())
    _max = max(text_feats.max(), toy_input_feats.max(), gt_output_feats.max(),
               diff.max())
    X = 512
    Y = 300
    images = []
    plt.figure(figsize=(30, 30))
    f, axarr = plt.subplots(2, 2)
    images.append(axarr[0, 1].imshow(text_feats[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[0], ax=axarr[0, 1])
    axarr[0, 1].set_title('Original image')
    images.append(axarr[1, 0].imshow(pred_output_ft[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[1], ax=axarr[1, 0])
    axarr[1, 0].set_title('Predicted features')
    images.append(axarr[1, 1].imshow(gt_output_feats[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[2], ax=axarr[1, 1])
    axarr[1, 1].set_title('Reverse mapping from \n reverse mapping')
    images.append(axarr[0, 0].imshow(diff[:Y, :X], vmin=_min, vmax=_max))
    plt.colorbar(images[3], ax=axarr[0, 0])
    axarr[0, 0].set_title('Difference between reverse \n mapping and prediction')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone', type=str, help='movie name')
    parser.add_argument('--direction', default='m2b', type=str, help='m2b or b2m')
    parser.add_argument('--loss_type', default='GT', type=str, help='GT or R')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, help='kernel type')
    parser.add_argument('--exp_info', type=str)
    parser.add_argument('--blur', type=str)

    args = parser.parse_args()
    
    # Define parameters
    input_size = 1
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 1
    num_epochs = 100000
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 512
    direction = args.direction
    loss_type = args.loss_type
    kernel_type = args.kernel_type
    blur = True if args.blur == 'y' else False

    # Tensorboard summary writer
    exp_name = f"train_{direction}_kernel_{kernel_type}_loss_{loss_type}_{args.exp_info}_try_{args.try_num}"
    # exp_name = f"train_{direction}_kernel_{kernel_type}_loss_{loss_type}_noSigmoid_weightdecay_{weight_decay}_h1_{hidden_size1}_h2_{hidden_size2}_lr_{lr}_batchsize_{batch_size}_try_{args.try_num}"
    writer = SummaryWriter(log_dir="runs/" + exp_name)

    if th.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu:0'
    # Get feats
    image_feats = np.load(f"data/{args.movie}/image_features.npy")
    text_feats = np.load(f"data/{args.movie}/text_features.npy")
    
    # Transform to tensors
    image_feats = th.FloatTensor(image_feats).T # shape (512, Nm)
    text_feats = th.FloatTensor(text_feats).T # shape (512, Nb)
    
    # Normalize
    image_feats /= image_feats.norm(dim=0, keepdim=True)
    text_feats /= text_feats.norm(dim=0, keepdim=True)
    
    # Get lens
    len_text = text_feats.shape[1]
    len_image = image_feats.shape[1]
    
    # Define input feats
    
    if direction == 'm2b':
        input_feats = image_feats
        len_input = len_image
        output_feats = text_feats.to(device)
        len_output = len_text
    else: 
        input_feats = text_feats
        len_input = len_text
        output_feats = image_feats
        len_output = len_image

    # Get input and output times
    input_times = th.FloatTensor(np.arange(len_input)).to(device)
    output_times = th.FloatTensor(np.arange(len_output)).to(device)
    
    # Scale input and output times to [0,1]
    input_times_scaled = input_times / (len_input - 1)
    output_times_scaled = output_times / (len_output - 1)
    
    # Get f times: from input -> output (shape Ni)
    f_times = input_times * (len_output - 1) / (len_input - 1)
    f_times = f_times.to(device)
    # Get invf times: from output -> input (shape No)
    invf_times = output_times * (len_input - 1) / (len_output - 1)
    invf_times = invf_times.to(device)
    
    # Get toy input feats: output_feats -> toy_input_feats
    # we need the reverse mapping input -> output
    toy_input_feats = reverse_mapping(output_feats, f_times, kernel_type, device).to(device)

    # from toy_input_feats to gt_output_feats (input -> output)
    gt_output_feats = reverse_mapping(toy_input_feats, invf_times, kernel_type, device).to(device)

    # Create times dataset and dataloader
    dataset = TimesDataset(output_times_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, sampler=None,
                #batch_sampler=None, num_workers=1, collate_fn=None)
    
    # Define model
    model = MLP(input_size, hidden_size1, hidden_size2, output_size, device=device)
    model= model.to(device)    
    # Define loss function
    loss_reconstruction = ReconstructionLoss()
    loss_cosine = CosineDistanceLoss()
    loss_gt = GTDifLoss()

    # Define optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Begin training
    #losses = th.Tensor(size = (num_epochs,2))
    start_time = time.time()
    loss_prev = 0
    loss_now = 1000
    epoch = 0
    while epoch < 500: #abs(loss_prev - loss_now) > 1e-10:
        pred_invf_times_scaled = []
        index = []
        # Run times through alignment network (mlp)
        for i, batch in enumerate(tqdm(dataloader)):
            batch, idx = batch
            batch, idx = batch.to(device), idx.to(device)
            invf_output = model.forward(batch.unsqueeze(1))
            pred_invf_times_scaled.append(invf_output)
            index.append(idx)

        # outputs are between 0 and 1 and have shape (No)
        pred_invf_times_scaled = th.cat(pred_invf_times_scaled, dim=0).squeeze(1) # shape No
        index = th.cat(index, dim=0)
        #pred_invf_times_scaled = pred_invf_times_scaled[index]
        
        # re-scale to 0 len_output -1
        pred_invf_times = pred_invf_times_scaled * (len_input - 1) # shape No

        # do inverse warping
        pred_output_feats = reverse_mapping(toy_input_feats, pred_invf_times, kernel_type, device).to(device)

        # Compute reconstruction loss
        lossR = loss_reconstruction(output_feats, pred_output_feats.to(device))
        #lossCD = loss_cosine(output_feats, pred_output_feats)
        lossGT = loss_gt(pred_invf_times_scaled, invf_times.to(device) / (len_input - 1))

        # Write to tensorboard
        writer.add_scalar("LossR/train", lossR, epoch)
        writer.add_scalar("LossGT/train", lossGT, epoch)
        #writer.add_scalar("LossCD/train", lossCD, epoch)
    
               
        # Backpropagate and update losses
        loss_prev = loss_now
        if loss_type == "GT":
            lossGT.backward()
            loss_now = lossGT
        elif loss_type == 'R':
            lossR.backward()
            loss_now = lossR
        else:
            lossCD.backward()
            loss_now = lossCD

        # Optimizer step
        optimizer.step()
        print(f"Epoch {epoch} loss: {lossGT}")
        
        # Save losses
        #losses[epoch][0] = lossR.detach()
        #losses[epoch][1] = lossGT.detach()
        
        # Only every 5 epochs, visualize images and mapping
        if epoch%5 == 0:
            # Define movie time segment to visualize
            i_len = 0
            ii_len = 100
            o_len = int(np.ceil(i_len/len_input*len_output))
            oo_len = int(np.ceil(ii_len/len_input*len_output))

            # Define feature segment to visualize
            f_len = 100
            # # Visualize movie
            #
            #
            # # Visualize predicted output feats
            # writer.add_image(f"Pred output feats {o_len}:{oo_len}", pred_output_feats[:f_len, o_len:oo_len], epoch, dataformats='HW')
            #
            # # Visualize difference of pred output feats and output feats
            # diff_pred_output_feats = pred_output_feats[:f_len, o_len:oo_len] - output_feats[:f_len, o_len:oo_len]
            # writer.add_image(f"Pred difference {o_len}:{oo_len}", diff_pred_output_feats, epoch, dataformats='HW')
            
            
            # Visualize mapping
            pred_invf_times_copy = pred_invf_times.clone().detach()
            plot_buf = get_plot(output_times.cpu(), pred_invf_times_copy.cpu(), invf_times.cpu())
            plot_buf_img = plot_diff(text_feats.data.numpy(), pred_output_feats.data.numpy(), gt_output_feats.data.numpy())
            
            image, vis = PIL.Image.open(plot_buf), PIL.Image.open(plot_buf_img)
            image, vis = ToTensor()(image), ToTensor()(vis)
            writer.add_image('Mapping', image, epoch)
            writer.add_image('Visualization', vis, epoch)
            plot_buf.close()
            plot_buf_img.close()
        
        
        epoch += 1
    
    end_time = time.time()
    writer.flush()
    writer.close()

    save_path = f"outputs/{args.movie}"
    #losses = losses.detach().numpy()
    #np.save(f"{save_path}/{exp_name}_loss.npy", losses)
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
 
