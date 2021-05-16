import numpy as np
import torch as th
import ipdb
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import time
import wandb
import numpy as np
import os
from network.mlp import MLP, positional_encoding
from warping.inverse_warping import reverse_mapping
from loss.losses import CosineDistanceLoss
from loss.losses import GTDifLoss
from utils import get_plot, plot_diff, visualize_input, plot_grad
import json
from make_gaussain_pyramide import get_image_pyramid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone', type=str, help='movie name')
    parser.add_argument('--direction', default='m2b', type=str, help='m2b or b2m')
    parser.add_argument('--gt_loss', default=1., type=float, help='weighting of gt loss')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, default='linear', help='kernel type')
    parser.add_argument('--pos_encoding', type=int, default=6, help='Number of encoding functions used to compute a positional encoding')
    parser.add_argument('--print_every', type=int, default=20, help='kernel type')
    parser.add_argument('--exp_info', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_image_pyramid_levels', type=int, default=5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--GP', type=float, default=0.)
    parser.add_argument('--R', type=float, default=0.)
    parser.add_argument('--GTD', type=float, default=0.)

    args = parser.parse_args()
    # Define parameters
    input_size = 1 + 2*args.pos_encoding
    output_size = 1
    num_epochs = 100000
    lr = args.lr
    weight_decay = 1e-4
    direction = args.direction
    kernel_type = args.kernel_type

    # Tensorboard summary writer
    exp_name = f"{direction}_{args.exp_info}_PE{args.pos_encoding}_GP{args.GP}_R{args.R}_GTD{args.GTD}_{args.movie}"
    wandb.init(project="book-movie-warping", entity="the-dream-team")
    wandb.run.name = exp_name
    wandb.config.update(args)

    use_pseudo_gt_dialog = False
    if args.GTD > 0.:
        use_pseudo_gt_dialog = True

    if th.cuda.is_available():
        device = 'cuda:{}'.format(args.cuda)
    else:
        device = 'cpu:0'

    # Get feats
    image_feats = np.load(f"data/{args.movie}/image_features.npy")
    text_feats = np.load(f"data/{args.movie}/text_features.npy")
    book_len = text_feats.shape[0]
    movie_len = image_feats.shape[0]

    # # Get dialog times
    # dialog_times = np.load(f'data/{args.movie}/dialog_times_dict.npy',
    #                        allow_pickle=True).item()
    #
    # # get visual sentences index
    # all_sentences = np.arange(book_len)
    # include_idx = set(dialog_times['dialog_book_times'])
    # mask = np.array([(i in include_idx) for i in range(book_len)])
    # visual_sentences = all_sentences[~mask]

    # Transform to tensors
    image_feats = th.FloatTensor(image_feats).T  # shape (512, Nm)
    text_feats = th.FloatTensor(text_feats).T  # shape (512, Nb)

    # Normalize
    image_feats /= image_feats.norm(dim=0, keepdim=True)
    text_feats /= text_feats.norm(dim=0, keepdim=True)

    # Get GT dictionary
    gt_dict = np.load(f"data/{args.movie}/gt_mapping.npy", allow_pickle=True)
    if use_pseudo_gt_dialog:
        gt_dict_dialog = np.load(f"data/{args.movie}/gt_dialog_matches.npy")
    rng = np.random.default_rng(2021)
    train = sorted(rng.choice(range(len(gt_dict)), len(gt_dict)//2+1, False))
    val = [i for i in range(len(gt_dict)) if i not in train]
    if args.direction == 'm2b':
        gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]
        org_len_input, org_len_output = movie_len, book_len
        org_input_feats, org_output_feats = image_feats.to(device), text_feats.to(device)
    else:
        gt_dict = [np.array([i['movie_ind'] for i in gt_dict]), np.array([i['book_ind'] for i in gt_dict])]
        gt_dict_dialog = [gt_dict_dialog[1], gt_dict_dialog[0]]
        org_len_input, org_len_output = book_len, movie_len
        org_input_feats, org_output_feats = text_feats.to(device), image_feats.to(device)


    # Get image pyramids
    text_pyramid = get_image_pyramid(text_feats)
    # text_pyramid_content = get_image_pyramid(text_feats[:, visual_sentences])
    image_pyramid = get_image_pyramid(image_feats)


    # Define model
    model = MLP(input_size, device=device) #hidden_size1, hidden_size2, output_size, device=device)
    model = model.to(device)
    if os.path.exists(args.resume):
        model.load_state_dict(th.load(args.resume))
        print(f'Model weights initialized from: {args.resume}')

    # Log model training
    wandb.watch(model, log="all")

    # loss_reconstruction = ReconstructionLoss()
    loss_rec = CosineDistanceLoss(device=device)
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


        # Define input feats
        if direction == 'm2b':
            input_feats = th.FloatTensor(image_pyramid[level]).to(device)
            output_feats = th.FloatTensor(text_pyramid[level]).to(device)
            len_input = len_image
            len_output = len_text
        else:
            input_feats = th.FloatTensor(text_pyramid[level]).to(device)
            output_feats = th.FloatTensor(image_pyramid[level]).to(device)
            len_input = len_text
            len_output = len_image

        # Get input and output times
        level_input_times = th.FloatTensor(np.arange(len_input)).to(device)
        level_output_times = th.autograd.Variable(th.FloatTensor(np.arange(len_output)).to(device), requires_grad=True)
        org_input_times = th.FloatTensor(np.arange(org_len_input)).to(device)
        org_output_times = th.autograd.Variable(th.FloatTensor(np.arange(org_len_output)).to(device), requires_grad=True)

        # Scale input and output times to [0,1]
        level_input_times_scaled = level_input_times / (len_input - 1)
        level_output_times_scaled = level_output_times / (len_output - 1)
        org_input_times_scaled = org_input_times / (org_len_input - 1)
        org_output_times_scaled = org_output_times / (org_len_output - 1)

        for i in range(args.num_epochs): # epoch < 500:
            inp1 = positional_encoding(level_output_times_scaled.unsqueeze(1), num_encoding_functions=args.pos_encoding)
            inp2 = positional_encoding(org_output_times_scaled.unsqueeze(1), num_encoding_functions=args.pos_encoding)
            pred_invf_times_scaled = model.forward(inp1).squeeze()
            org_pred_invf_times_scaled = model.forward(inp2).squeeze()
            # re-scale to 0 len_output -1
            pred_invf_times = pred_invf_times_scaled * (len_input - 1) # shape No
            org_pred_invf_times = org_pred_invf_times_scaled * (org_len_input - 1) # shape No

            gradspred, = th.autograd.grad(org_pred_invf_times, org_output_times,
                                       grad_outputs=org_pred_invf_times.data.new(org_pred_invf_times.shape).fill_(1),
                                       retain_graph=True, create_graph=True)
            grad_penalty = -gradspred[gradspred < 0].mean()
            pred_output_feats = reverse_mapping(input_feats, pred_invf_times.squeeze(), kernel_type).to(device)
            lossR = loss_rec(output_feats, pred_output_feats.to(device))

            lossGT = th.nn.functional.l1_loss(org_pred_invf_times_scaled[gt_dict[0][train]], th.LongTensor(gt_dict[1][train]).to(device)/org_len_input)
            lossGT_val = th.nn.functional.l1_loss(org_pred_invf_times_scaled[gt_dict[0][val]], th.LongTensor(gt_dict[1][val]).to(device)/org_len_input)
            lossGTD = th.nn.functional.l1_loss(org_pred_invf_times_scaled[gt_dict_dialog[0]], th.FloatTensor(gt_dict_dialog[1]).to(device)/org_len_input) if use_pseudo_gt_dialog else 0.

            if level == 0:
                fine_pred_output_feats = pred_output_feats
            else:
                fine_pred_output_feats = reverse_mapping(org_input_feats, org_pred_invf_times.squeeze(), kernel_type).to(device)
            gt_sim_score = th.mul(fine_pred_output_feats[:, gt_dict[0]], org_output_feats[:, gt_dict[0]]).sum(0).mean()
            gt_sim_score_dialog = th.mul(fine_pred_output_feats[:, gt_dict_dialog[0]],
                                         org_output_feats[:, gt_dict_dialog[0]]).sum(0).mean() if use_pseudo_gt_dialog else 0.
            score_fine_scale = th.mul(fine_pred_output_feats, org_output_feats).sum(0).mean()
            # Write to wandb
            wandb.log({'epoch': epoch,
                       'reconstruction_loss_content': -lossR,
                       'ground_truth_loss': lossGT,
                       'ground_truth_loss_dialog': lossGTD,
                       'gt_similarity_score': gt_sim_score,
                       'gt_similarity_score_dialog': gt_sim_score_dialog,
                       'similarity_score_all': score_fine_scale,
                       'ground_truth_loss_validation': lossGT_val,
                       'grad_penalty': grad_penalty, 'lr': lr,
                       })

            # Backpropagate and update losses
            loss_prev = loss_now
            optimizer.zero_grad()

            if loss_now > loss_prev:
                lr *= 0.1

            loss_ = args.gt_loss*lossGT + args.R*lossR + args.GTD*lossGTD + args.GP*grad_penalty
            loss_.backward(retain_graph=True)
            loss_now = loss_

            # Optimizer step
            optimizer.step()

            # Only every 5 epochs, visualize images and mapping
            if epoch % args.print_every == 0:
                print(f"Epoch {epoch} loss GT: {lossGT} loss R {lossR} loss GTD {lossGTD}")
                if epoch == 0:
                    # Visualize input and output
                    visualize_input(input_feats.cpu().data.numpy(), output_feats.cpu().data.numpy())

                # Visualize mapping
                get_plot(org_output_times.cpu().detach().numpy(), org_pred_invf_times.detach().cpu(), gt_dict,
                         split={'train': train, 'val': val}, gt_dict_dialog=gt_dict_dialog if use_pseudo_gt_dialog else None)
                plot_diff(input_feats.cpu().data.numpy(),
                          pred_output_feats.cpu().data.numpy(),
                          output_feats.cpu().data.numpy(), titles=['Input', 'Prediction', 'Output', 'Difference'])
                plot_grad(gradspred.cpu().detach().numpy())
            epoch += 1

    
    end_time = time.time()
    save_path = f"outputs/{args.movie}"
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
