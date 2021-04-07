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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone', type=str, help='movie name')
    parser.add_argument('--direction', default='m2b', type=str, help='m2b or b2m')
    parser.add_argument('--loss_type', default='GT', type=str, help='GT or R')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, help='kernel type')
    args = parser.parse_args()
    
    # Define parameters
    input_size = 1
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 1
    num_epochs = 100000
    lr = 1e-4
    weight_decay = 1e-5
    batch_size = 512
    direction = args.direction
    loss_type = args.loss_type
    kernel_type = args.kernel_type

    # Tensorboard summary writer
    exp_name = f"train_{direction}_kernel_{kernel_type}_loss_{loss_type}_SHUFFLE_try_{args.try_num}"
    # exp_name = f"train_{direction}_kernel_{kernel_type}_loss_{loss_type}_noSigmoid_weightdecay_{weight_decay}_h1_{hidden_size1}_h2_{hidden_size2}_lr_{lr}_batchsize_{batch_size}_try_{args.try_num}"
    writer = SummaryWriter(log_dir="runs/" + exp_name)

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
        output_feats = text_feats
        len_output = len_text
    else: 
        input_feats = text_feats
        len_input = len_text
        output_feats = image_feats
        len_output = len_image

    # Get input and output times
    input_times = th.FloatTensor(np.arange(len_input)) 
    output_times = th.FloatTensor(np.arange(len_output))
    
    # Scale input and output times to [0,1]
    input_times_scaled = input_times / (len_input - 1)
    output_times_scaled = output_times / (len_output - 1)
    
    # Get f times: from input -> output (shape Ni)
    f_times = input_times * (len_output - 1) / (len_input - 1)
    
    # Get invf times: from output -> input (shape No)
    invf_times = output_times * (len_input - 1) / (len_output - 1)
    
    # Get toy input feats: output_feats -> toy_input_feats
    # we need the reverse mapping input -> output
    toy_input_feats = reverse_mapping(output_feats, f_times, kernel_type)

    # from toy_input_feats to gt_output_feats (input -> output)
    gt_output_feats = reverse_mapping(toy_input_feats, invf_times, kernel_type)

    # Create times dataset and dataloader
    dataset = TimesDataset(output_times_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
                batch_sampler=None, num_workers=1, collate_fn=None)
    
    # Define model
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)
    
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
    while abs(loss_prev - loss_now) > 1e-10:
        pred_invf_times_scaled = []
        index = []
        # Run times through alignment network (mlp)
        for i, batch in enumerate(tqdm(dataloader)):
            batch, idx = batch
            invf_output = model.forward(batch.unsqueeze(1))
            pred_invf_times_scaled.append(invf_output)
            index.append(idx)

        # outputs are between 0 and 1 and have shape (No)
        pred_invf_times_scaled = th.cat(pred_invf_times_scaled, dim=0).squeeze(1) # shape No
        index = th.cat(index, dim=0)
        pred_invf_times_scaled = pred_invf_times_scaled[index]
        
        # re-scale to 0 len_output -1
        pred_invf_times = pred_invf_times_scaled * (len_input - 1) # shape No

        # do inverse warping
        pred_output_feats = reverse_mapping(toy_input_feats, pred_invf_times, kernel_type)

        # Compute reconstruction loss
        lossR = loss_reconstruction(output_feats, pred_output_feats)
        #lossCD = loss_cosine(output_feats, pred_output_feats)
        lossGT = loss_gt(pred_invf_times_scaled, invf_times / (len_input - 1))

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
            # Visualize movie
            if epoch == 0:
                # Visualize toy input feats
                writer.add_image(f"Toy input feats {i_len}:{ii_len}", toy_input_feats[:f_len, i_len:ii_len].T, 0,  dataformats='HW')

                # Visualize output feats
                writer.add_image(f"Output feats {o_len}:{oo_len}", output_feats[:f_len, o_len:oo_len].T, 0, dataformats='HW')
                
                # Visualize gt pred output feats
                gt_output_feats = gt_output_feats[:f_len, o_len:oo_len]
                writer.add_image(f"GT predicted output feats {o_len}:{oo_len}", gt_output_feats.T, 0, dataformats='HW')
                
                # Visualize difference of gt pred output feats and output feats
                diff_gt_output_feats = gt_output_feats[:f_len, o_len:oo_len] - output_feats[:f_len, o_len:oo_len]
                writer.add_image(f"GT difference {o_len}:{oo_len}", diff_gt_output_feats.T, 0, dataformats='HW')

                
            # Visualize predicted output feats
            writer.add_image(f"Pred output feats {o_len}:{oo_len}", pred_output_feats[:f_len, o_len:oo_len].T, epoch, dataformats='HW')
            
            # Visualize difference of pred output feats and output feats
            diff_pred_output_feats = pred_output_feats[:f_len, o_len:oo_len] - output_feats[:f_len, o_len:oo_len]
            writer.add_image(f"Pred difference {o_len}:{oo_len}", diff_pred_output_feats.T, epoch, dataformats='HW')
            
            
            # Visualize mapping
            pred_invf_times_copy = pred_invf_times.clone().detach()
            plot_buf = get_plot(output_times, pred_invf_times_copy, invf_times)
            
            image = PIL.Image.open(plot_buf)
            image = ToTensor()(image)
            writer.add_image('Mapping', image, epoch)
            plot_buf.close()
        
        
        epoch += 1
    
    end_time = time.time()
    writer.flush()
    writer.close()

    save_path = f"outputs/{args.movie}"
    #losses = losses.detach().numpy()
    #np.save(f"{save_path}/{exp_name}_loss.npy", losses)
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
 
