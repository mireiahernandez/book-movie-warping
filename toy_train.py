import numpy as np
import torch as th
import ipdb
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import time


from network.mlp import MLP
from warping.inverse_warping import reverse_mapping
from torch.nn.functional import grid_sample
from loss.losses import ReconstructionLoss, CosineDistanceLoss
from loss.losses import GTDifLoss, GTNormLoss
from utils import get_plot, plot_diff, visualize_input

from torch.utils.tensorboard import SummaryWriter
import wandb

import numpy as np
import os
import scipy.signal as signal

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class TimesDataset(Dataset):
    def __init__(self, times):
        self.times = times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.times[idx], idx



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone', type=str, help='movie name')
    parser.add_argument('--direction', default='m2b', type=str, help='m2b or b2m')
    parser.add_argument('--gt_loss', default=0.0, type=float, help='weighting of gt loss')
    parser.add_argument('--rec_loss', default=0.0, type=float, help='weighting of rec loss')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, help='kernel type')
    parser.add_argument('--print_every', type=int, default=20, help='kernel type')
    parser.add_argument('--exp_info', type=str)
    parser.add_argument('--blur', default='n', type=str, help='y for blur else not using blur')
    parser.add_argument('--h1', type=int, default=64, help='hidden dim 1')
    parser.add_argument('--h2', type=int, default=32, help='hidden dim 2')
    args = parser.parse_args()



    # Define parameters
    input_size = 1
    hidden_size1 = args.h1
    hidden_size2 = args.h2
    output_size = 1
    num_epochs = 100000
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 512
    direction = args.direction
    kernel_type = args.kernel_type
    blur = True if args.blur == 'y' else False

    # Tensorboard summary writer
    exp_name = f"train_{direction}_kernel_{kernel_type}_loss_{args.exp_info}_try_{args.try_num}_h1h2{args.h1}_{args.h2}"
    writer = SummaryWriter(log_dir="runs/" + exp_name)
    wandb.init(project="book-movie-warping", entity="the-dream-team")
    wandb.run.name = exp_name
    wandb.config.update(args)

    if th.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu:0'

    # Get feats
    image_feats = np.load(f"data/{args.movie}/image_features.npy")
    text_feats = np.load(f"data/{args.movie}/text_features.npy")

    # Transform to tensors and blur (optional)
    if blur and direction == 'm2b':
        kernel = np.array([[1, 4, 6, 4, 1]])/16
        text_feats = signal.convolve2d(text_feats.T, kernel, mode='valid', boundary='wrap')
        text_feats = th.FloatTensor(text_feats)
        image_feats = th.FloatTensor(image_feats).T
    elif blur and direction == 'b2m':
        text_feats = th.FloatTensor(text_feats).T
        kernel = np.array([[1, 4, 6, 4, 1]])/16
        image_feats = signal.convolve2d(image_feats.T, kernel, mode='valid', boundary='wrap')
        image_feats = th.FloatTensor(image_feats)
    else:
        image_feats = th.FloatTensor(image_feats).T  # shape (512, Nm)
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
        output_feats = image_feats.to(device)
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
    toy_input_feats = reverse_mapping(output_feats, f_times, kernel_type).to(device)

    # from toy_input_feats to gt_output_feats (input -> output)
    gt_output_feats = reverse_mapping(toy_input_feats, invf_times, kernel_type).to(device)

    # Create times dataset and dataloader
    dataset = TimesDataset(output_times_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, sampler=None,
                #batch_sampler=None, num_workers=1, collate_fn=None)
    
    # Define model
    model = MLP(input_size, hidden_size1, hidden_size2, output_size, device=device)
    model = model.to(device)

    # Log model training
    wandb.watch(model, log="all")

    # Define loss function
    loss_reconstruction = ReconstructionLoss()
    loss_cosine = CosineDistanceLoss()
    loss_gt = GTDifLoss()

    # Define optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Begin training
    start_time = time.time()
    loss_prev = 0
    loss_now = 1000
    epoch = 0
    while epoch < 500 : #abs(loss_prev - loss_now) > 1e-10 and epoch<500:
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
        # index = th.cat(index, dim=0).argsort()
        # pred_invf_times_scaled = pred_invf_times_scaled[index]
        
        # re-scale to 0 len_output -1
        pred_invf_times = pred_invf_times_scaled * (len_input - 1) # shape No

        # do inverse warping
        pred_output_feats = reverse_mapping(toy_input_feats, pred_invf_times, kernel_type).to(device)

        # Compute reconstruction loss
        lossR = loss_reconstruction(output_feats, pred_output_feats.to(device))

        #lossCD = loss_cosine(output_feats, pred_output_feats)
        lossGT = loss_gt(pred_invf_times_scaled, invf_times.to(device) / (len_input - 1))

        # Write to tensorboard
        # writer.add_scalar("LossR/train", lossR, epoch)
        # writer.add_scalar("LossGT/train", lossGT, epoch)
        # #writer.add_scalar("LossCD/train", lossCD, epoch)
        wandb.log({'epoch': epoch,
                   'reconstruction_loss': lossR,
                   'ground_truth_loss': lossGT})
    
               
        # Backpropagate and update losses
        loss_prev = loss_now
        optimizer.zero_grad()

        if loss_now > loss_prev:
            lr *= 0.1
        # if loss_type == "GT":
        #     lossGT.backward()
        #     loss_now = lossGT
        # elif loss_type == 'R':
        #     lossR.backward()
        #     loss_now = lossR
        # else:
        #     lossCD.backward()
        #     loss_now = lossCD
        loss_ = args.gt_loss * lossGT + args.rec_loss * lossR
        loss_.backward()
        loss_now = loss_


        # Optimizer step
        optimizer.step()
        print(f"Epoch {epoch} loss: {lossGT}")

        
        # Only every 5 epochs, visualize images and mapping
        if epoch % args.print_every == 0:
            if epoch == 0:
                # Visualize input and output
                visualize_input(input_feats.cpu().data.numpy(), output_feats.cpu().data.numpy())
            
            # Visualize mapping
            get_plot(output_times.cpu(), pred_invf_times.detach().cpu(), invf_times.cpu())
            plot_diff(text_feats.cpu().data.numpy(),
                                     pred_output_feats.cpu().data.numpy(),
                                     gt_output_feats.cpu().data.numpy())


        
        
        epoch += 1
    
    end_time = time.time()


    save_path = f"outputs/{args.movie}"
    #losses = losses.detach().numpy()
    #np.save(f"{save_path}/{exp_name}_loss.npy", losses)
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
    # wandb.save('model.h5')
 
