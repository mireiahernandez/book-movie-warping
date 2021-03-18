import numpy as np
import torch as th
import ipdb
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import time

from network.mlp import MLP
from warping.forward_warping import forward_warping
from torch.nn.functional import grid_sample
from loss.losses import ReconstructionLoss


class TimesDataset(Dataset):
    def __init__(self, times):
        self.times = times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.times[idx]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', type=str, help='movie name')
    parser.add_argument('--run_name', type=str, help='run name')
    args = parser.parse_args()
    
    # Get features
    image_features = np.load(f"data/{args.movie}/image_features.npy")
    text_features = np.load(f"data/{args.movie}/text_features.npy")
    
    # Transform to tensors
    image_features = th.FloatTensor(image_features).T
    text_features = th.FloatTensor(text_features).T
    
    # Get input times
    len_text = text_features.shape[1]
    len_image = image_features.shape[1]
    
    input_times = np.array(list(range(0, len_image))) / len_image
    input_times = th.FloatTensor(input_times)
    
    # Create times dataset and dataloader
    dataset = TimesDataset(input_times)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=20, collate_fn=None)
    
    # Define MLP model
    input_size = 1
    hidden_size = 36
    output_size = 1
    model = MLP(input_size, hidden_size, output_size)
    loss_fn = ReconstructionLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100
    losses = th.Tensor(size = (num_epochs,))
    print(f"lr: {0.01}")
    start_time = time.time()
    for epoch in range(num_epochs):
        outputs = []
        # Run times through alignment network (mlp)
        for i, batch in enumerate(tqdm(dataloader)):
            output = model.forward(batch.unsqueeze(1))
            outputs.append(output)
        # outputs are between 0 and 1 and have shape (Nmovie, 1)
        outputs = th.cat(outputs, dim=0)

        # scale outputs so they range between -1 and 1
        outputs_scaled = 2 * outputs - 1
        
        # Calculate grid of shape (Hout, Wout, 2) (512, N_movie)
        grid = th.zeros(size=(1, 512, len_image, 2))
        for i in range(512):
            grid[:, i, :, 0] = 2 * i / 512 - 1
            grid[:, i, :, 1] = outputs_scaled.squeeze(1)

        # Do inverse image warping
        warped_text_features = grid_sample(text_features.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                    padding_mode='zeros', align_corners=None)
        warped_text_features = warped_text_features.squeeze(0).squeeze(0)
        
        # Compute reconstruction loss
        loss = loss_fn(image_features, warped_text_features)
        losses[epoch] = loss
        # Backpropagate
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        print(f"Epoch {epoch} loss: {loss}")
    end_time = time.time()
    ipdb.set_trace()
    save_path = f"outputs/{args.movie}"
    losses = losses.detach().numpy()
    np.save(f"{save_path}/{args.run_name}_loss.npy", losses)
    th.save(model.state_dict(), f"{save_path}/{args.run_name}_model.pt")

