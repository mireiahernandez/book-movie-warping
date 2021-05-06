import numpy as np
import torch as th
import ipdb
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import os
from network.mlp import MLP
from warping.inverse_warping import reverse_mapping
from loss.losses import ReconstructionLoss, CosineDistanceLoss
from loss.losses import GTDifLoss, GTNormLoss, SimilarityDialog
from utils import get_plot, plot_diff, visualize_input
import json
from make_gaussain_pyramide import get_image_pyramid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TimesDataset(Dataset):
    def __init__(self, times):
        self.times = times

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return self.times[idx], idx



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone_GT', type=str, help='movie name')
    parser.add_argument('--direction', default='m2b', type=str, help='m2b or b2m')
    parser.add_argument('--gt_loss', default=0.0, type=float, help='weighting of gt loss')
    parser.add_argument('--rec_loss', default=0.0, type=float, help='weighting of rec loss')
    parser.add_argument('--recd_loss', default=0.0, type=float, help='weighting of rec loss for dialog')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, default='linear', help='kernel type')
    parser.add_argument('--print_every', type=int, default=20, help='kernel type')
    parser.add_argument('--exp_info', type=str)
    parser.add_argument('--blur', default='n', type=str, help='y for blur else not using blur')
    parser.add_argument('--h1', type=int, default=64, help='hidden dim 1')
    parser.add_argument('--h2', type=int, default=32, help='hidden dim 2')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_image_pyramid_levels', type=int, default=5)

    args = parser.parse_args()
    # Define parameters
    input_size = 1
    hidden_size1 = args.h1
    hidden_size2 = args.h2
    output_size = 1
    num_epochs = 100000
    lr = args.lr
    weight_decay = 1e-4
    batch_size = 7000
    direction = args.direction
    kernel_type = args.kernel_type
    blur = True if args.blur == 'y' else False

    # Tensorboard summary writer
    exp_name = f"{direction}_{args.exp_info}_try_{args.try_num}"
    writer = SummaryWriter(log_dir="runs/" + exp_name)
    wandb.init(project="book-movie-warping", entity="the-dream-team")
    wandb.run.name = exp_name
    wandb.config.update(args)

    if th.cuda.is_available():
        device = 'cuda:{}'.format(args.cuda)
    else:
        device = 'cpu:0'

    # Get feats
    image_feats = np.load(f"data/{args.movie}/image_features.npy")
    text_feats = np.load(f"data/{args.movie}/text_features.npy")
    book_len = text_feats.shape[0]
    movie_len = image_feats.shape[0]

    # Get dialog feats
    dialog_book = np.load(f'data/{args.movie}/dialog_book_features.npy')
    dialog_movie = np.load(f'data/{args.movie}/dialog_cc_features.npy')
    dialog_times = np.load(f'data/{args.movie}/dialog_times_dict.npy',
                           allow_pickle=True).item()

    # Transform to tensors
    image_feats_dialog = th.FloatTensor(dialog_movie).T  # shape (512, Nm)
    text_feats_dialog = th.FloatTensor(dialog_book).T  # shape (512, Nb)

    # Normalize
    image_feats_dialog /= image_feats_dialog.norm(dim=0, keepdim=True)
    text_feats_dialog /= text_feats_dialog.norm(dim=0, keepdim=True)

    # get visual sentences index
    all_sentences = np.arange(book_len)
    include_idx = set(dialog_times['dialog_book_times'])
    mask = np.array([(i in include_idx) for i in range(book_len)])
    visual_sentences = all_sentences[~mask]

    # Get GT dictionary
    gt_dict = json.load(open(f"data/{args.movie}/gt_mapping.json", 'r'))
    if args.direction == 'm2b':
        gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]
    else:
        gt_dict = [np.array([i['movie_ind'] for i in gt_dict]), np.array([i['book_ind'] for i in gt_dict])]



    # Transform to tensors
    image_feats = th.FloatTensor(image_feats).T  # shape (512, Nm)
    text_feats = th.FloatTensor(text_feats).T # shape (512, Nb)
    
    # Normalize
    image_feats /= image_feats.norm(dim=0, keepdim=True)
    text_feats /= text_feats.norm(dim=0, keepdim=True)

    # Get image pyramids
    text_pyramid = get_image_pyramid(text_feats)
    text_pyramid_content = get_image_pyramid(text_feats[:, visual_sentences])
    image_pyramid = get_image_pyramid(image_feats)


    # Define model
    model = MLP(input_size, hidden_size1, hidden_size2, output_size, device=device)
    model = model.to(device)

    # Log model training
    wandb.watch(model, log="all")

    # loss_reconstruction = ReconstructionLoss()
    loss_rec = CosineDistanceLoss(device=device)
    loss_gt = GTDifLoss()
    similarity_dialog = SimilarityDialog()
    # Define optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Begin training
    start_time = time.time()
    loss_prev, loss_now = 0, 1000
    epoch = 0

    # coarse to fine alignment
    for level in reversed(range(args.num_image_pyramid_levels)):
        # Get lens
        len_text = text_pyramid[level].shape[1]
        len_image = image_pyramid[level].shape[1]

        # Get GT dictionary
        gt_dict = json.load(open(f"data/{args.movie}/gt_mapping.json", 'r'))
        gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]
        gt_dict[0] = gt_dict[0]//2**level
        gt_dict[1] = gt_dict[1]//2**level


        # Define input feats
        if direction == 'm2b':
            input_feats = th.FloatTensor(image_pyramid[level]).to(device)
            output_feats = th.FloatTensor(text_pyramid_content[level]).to(device)
            len_input = len_image
            len_output = len_text
            input_feats_dialog = th.FloatTensor(image_feats_dialog).to(device)
            output_feats_dialog = th.FloatTensor(text_feats_dialog).to(device)
            similarity_dialogs_lookup = input_feats_dialog.T @ output_feats_dialog

            # get indexes for f_theta for content
            index_level_visual_sentences = th.FloatTensor(np.array(visual_sentences[::2 ** level]) // 2 ** level).to(
                device)
            # mapping for dialogs in original time input
            v = sorted(list(set(np.arange(movie_len)[::2 ** level]).intersection(dialog_times['dialog_movie_times'])))
            # subsampled dialogs in coarser level input
            d_in = [dialog_times['dialog_movie_times'].index(i) for i in v]
            # mapping for dialogs in original time output
            v1 = sorted(list(set(np.arange(book_len)[::2 ** level]).intersection(dialog_times['dialog_book_times'])))
            # subsampled dialogs in coarser level output
            d_out = [dialog_times['dialog_book_times'].index(i) for i in v1]
            dialog_index_subsampled = th.FloatTensor(np.array(v1) // (2 ** level))
        else:
            input_feats = th.FloatTensor(text_pyramid_content[level]).to(device)
            output_feats = th.FloatTensor(image_pyramid[level]).to(device)
            len_input = len_text
            len_output = len_image
            len_input_dialog = len(dialog_book)
            input_feats_dialog = th.FloatTensor(text_feats_dialog).to(device)
            output_feats_dialog = th.FloatTensor(image_feats_dialog).to(device)
            similarity_dialogs_lookup = output_feats_dialog.T @ input_feats_dialog
            # get indexes for f_theta for content
            index_level_visual_sentences = th.FloatTensor(np.array(visual_sentences[::2 ** level]) // 2 ** level).to(
                device)
            # mapping for dialogs in original time output
            v = sorted(list(set(np.arange(movie_len)[::2 ** level]).intersection(dialog_times['dialog_movie_times'])))
            # subsampled dialogs in coarser level output
            d_out = [dialog_times['dialog_movie_times'].index(i) for i in v]
            # mapping for dialogs in original time input
            v1 = sorted(list(set(np.arange(book_len)[::2 ** level]).intersection(dialog_times['dialog_book_times'])))
            # subsampled dialogs in coarser level input
            d_in = [dialog_times['dialog_book_times'].index(i) for i in v1]
            dialog_index_subsampled = th.FloatTensor(np.array(v) // (2 ** level))


        # Get input and output times
        input_times = th.FloatTensor(np.arange(len_input)).to(device)
        output_times = th.FloatTensor(np.arange(len_output)).to(device)

        # Scale input and output times to [0,1]
        input_times_scaled = input_times / (len_input - 1)
        output_times_scaled = output_times / (len_output - 1)


        # Create times dataset and dataloader
        dataset = TimesDataset(output_times_scaled)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if level !=0:
            r = 100
        else:
            r = 500
        print(len_input, len_output)
        for i in range(r): # epoch < 500:
        #    print(r, level, 'num_epochs', 'level')          
            pred_invf_times_scaled = []
            index = []
            # Run times through alignment network (mlp)
            for i, batch in enumerate(dataloader):
                batch = batch[0].to(device)

                invf_output = model.forward(batch.unsqueeze(1))
                pred_invf_times_scaled.append(invf_output)


            # outputs are between 0 and 1 and have shape (No)
            pred_invf_times_scaled = th.cat(pred_invf_times_scaled, dim=0).squeeze(1) # shape No

            # re-scale to 0 len_output -1
            pred_invf_times = pred_invf_times_scaled * (len_input - 1) # shape No


            # Compute reconstruction loss
            if direction == 'm2b':
                pred_invf_times_content = pred_invf_times[index_level_visual_sentences.long()]
                v_content = None
                v_dialog = v
                pred_invf_times = pred_invf_times_scaled * (movie_len - 1)  # shape No
                pred_invf_times_dialog = pred_invf_times[dialog_index_subsampled.long()]
            else:
                # in 'b2m' all the times are visual times because they contain frames
                pred_invf_times_content = pred_invf_times
                v_content = index_level_visual_sentences
                v_dialog = v1
                pred_invf_times = pred_invf_times_scaled * (book_len - 1)  # shape No
                pred_invf_times_dialog = pred_invf_times[dialog_index_subsampled.long()]

            pred_output_feats = reverse_mapping(input_feats, pred_invf_times_content, kernel_type, v=v_content).to(device)
            lossR = loss_rec(output_feats, pred_output_feats.to(device))
            lossRD, lossRDL2 = similarity_dialog(similarity_dialogs_lookup[d_in, :][:, d_out], pred_invf_times_dialog,
                                       th.FloatTensor(v_dialog).to(device), device=device, temperature=100)

            # input are the predicted coordinates in movie where they should match to book,
            # output are matching coordinates in the book
            lossGT = th.nn.functional.l1_loss(pred_invf_times[gt_dict[0]], th.LongTensor(gt_dict[1]).to(device))
            #ipdb.set_trace()
            # metric = avg_distance_to_nearest_gt()
            # Write to wandb
            wandb.log({'epoch': epoch,
                       'reconstruction_loss_content': -lossR,
                       'ground_truth_loss': lossGT,
                       'reconstruction_loss_dialog': -lossRD,
                       # 'selfsupervised_loss': lossSS,
                       'lambda_gt': args.gt_loss, 'lambda_rec': args.rec_loss,
                       'lambda_recd': args.recd_loss, 'loss_dialog_dist': lossRDL2})

            # Backpropagate and update losses
            loss_prev = loss_now
            optimizer.zero_grad()

            if loss_now > loss_prev:
                lr *= 0.1

            loss_ = args.gt_loss * lossGT + args.rec_loss * lossR + args.recd_loss * lossRD + lossRDL2#args.clip_loss * lossCLIP + args.ss_loss * lossSS
            loss_.backward()
            loss_now = loss_


            # Optimizer step
            optimizer.step()
            #print(f"Epoch {epoch} loss GT: {lossGT} loss R {lossR} loss RDialog {lossRD}")


            # Only every 5 epochs, visualize images and mapping
            if epoch % args.print_every == 0:
                print(f"Epoch {epoch} loss GT: {lossGT} loss R {lossR} loss RDialog {lossRD} loss RDL2 {lossRDL2}")
                if epoch == 0:
                    # Visualize input and output
                    visualize_input(input_feats.cpu().data.numpy(), output_feats.cpu().data.numpy())

                # Visualize mapping

                get_plot(output_times.cpu(), pred_invf_times.detach().cpu(), gt_dict)
                plot_diff(input_feats.cpu().data.numpy(),
                          pred_output_feats.cpu().data.numpy(),
                          output_feats.cpu().data.numpy(), titles=['Input', 'Prediction', 'Output', 'Difference'])
            epoch += 1



    
    end_time = time.time()


    save_path = f"outputs/{args.movie}"
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
    # wandb.save('model.h5')
 
