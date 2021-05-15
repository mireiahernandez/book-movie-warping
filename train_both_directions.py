import torch as th
import ipdb
import argparse
import time
import wandb
import numpy as np
import os
from network.mlp import MLP, positional_encoding, MLP_2dir
from warping.inverse_warping import reverse_mapping
from loss.losses import CosineDistanceLoss
from utils import get_plot, plot_diff, visualize_input, plot_grad
from make_gaussain_pyramide import get_image_pyramid

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone', type=str, help='movie name')
    parser.add_argument('--gt_loss', default=0.0, type=float, help='weighting of gt loss')
    parser.add_argument('--rec_loss', default=0.0, type=float, help='weighting of rec loss')
    parser.add_argument('--gtd_loss', default=0.0, type=float, help='weighting of rec loss for dialog')
    parser.add_argument('--try_num', type=str, help='try number')
    parser.add_argument('--kernel_type', type=str, default='linear', help='kernel type')
    parser.add_argument('--pos_encoding', type=int, default=6, help='Number of encoding functions used to compute a positional encoding')
    parser.add_argument('--print_every', type=int, default=50, help='kernel type')
    parser.add_argument('--exp_info', default='', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_image_pyramid_levels', type=int, default=5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--loss_scaled', type=str, default='N')

    args = parser.parse_args()
    # Define parameters
    input_size = 1 + 2*args.pos_encoding
    output_size = 1
    num_epochs = 100
    lr = args.lr
    weight_decay = 1e-4
    kernel_type = args.kernel_type

    # Tensorboard summary writer
    exp_name = f"M2B2M_{args.exp_info}_try_{args.try_num}"
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
    org_book_len = text_feats.shape[0]
    org_movie_len = image_feats.shape[0]

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
    gt_dict_dialog = np.load(f"data/{args.movie}/gt_dialog_matches.npy")
    rng = np.random.default_rng(2021)
    train = sorted(rng.choice(range(len(gt_dict)), len(gt_dict)//2+1, False))
    val = [i for i in range(len(gt_dict)) if i not in train]
    gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]
    org_movie_feats, org_book_feats = image_feats.to(device), text_feats.to(device)



    # Get image pyramids
    text_pyramid = get_image_pyramid(text_feats)
    # text_pyramid_content = get_image_pyramid(text_feats[:, visual_sentences])
    image_pyramid = get_image_pyramid(image_feats)


    # Define model
    model = MLP_2dir(input_size, device=device, PE=args.pos_encoding) #hidden_size1, hidden_size2, output_size, device=device)
    model.to(device)
    if os.path.exists(args.resume):
        model.load_state_dict(th.load(args.resume))
        print(f'Model weights initialized from: {args.resume}')

    # Log model training
    wandb.watch(model, log="all")

    # loss_reconstruction = ReconstructionLoss()
    loss_rec = CosineDistanceLoss(device=device)
    l1_loss = th.nn.L1Loss()
    # Define optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Begin training
    start_time = time.time()
    loss_prev, loss_now = 0, 1000
    epoch = 0

    # coarse to fine alignment
    for level in reversed(range(args.num_image_pyramid_levels)):
        # Get lens
        len_book = text_pyramid[level].shape[1]
        len_movie = image_pyramid[level].shape[1]
        movie_feats = th.FloatTensor(image_pyramid[level]).to(device)
        book_feats = th.FloatTensor(text_pyramid[level]).to(device)

        # Get input and output times
        level_movie_times = th.autograd.Variable(th.FloatTensor(np.arange(len_movie)), requires_grad=True).to(device)
        level_book_times = th.autograd.Variable(th.FloatTensor(np.arange(len_book)), requires_grad=True).to(device)
        org_movie_times = th.autograd.Variable(th.FloatTensor(np.arange(org_movie_len)), requires_grad=True).to(device)
        org_book_times = th.autograd.Variable(th.FloatTensor(np.arange(org_book_len)), requires_grad=True).to(device)

        # Scale input and output times to [0,1]
        level_movie_times_scaled = level_movie_times / (len_movie - 1)
        level_book_times_scaled = level_book_times / (len_book - 1)
        org_movie_times_scaled = org_movie_times / (org_movie_len - 1)
        org_book_times_scaled = org_book_times / (org_book_len - 1)

        num_epochs = 150
        for i in range(num_epochs):
            # 1st forward pass on the coarse scale
            m1, b1, b2m_m2b_b, m2b_b2m_m = model.forward(level_movie_times_scaled, level_book_times_scaled)
            # 2nd forward pass on the fine scale
            org_m1, org_b1, org_b2m_m2b_b, org_m2b_b2m_m = model.forward(org_movie_times_scaled, org_book_times_scaled)
            org_b2m_m2b_b, org_m2b_b2m_m = org_b2m_m2b_b.squeeze(), org_m2b_b2m_m.squeeze()
            org_m1, org_b1 = org_m1.squeeze(), org_b1.squeeze()
            # re-scale to 0 len_output -1
            m1_org_range = m1 * (len_movie - 1) # shape No
            org_m1_org_range = org_m1 * (org_movie_len - 1) # shape No
            b1_org_range = b1 * (len_book - 1) # shape No
            org_b1_org_range = org_b1 * (org_book_len - 1) # shape No

            # CYCLE CONSISTENCY LOSS
            lossCC_b2m = l1_loss(org_movie_times_scaled, org_m2b_b2m_m)
            lossCC_m2b = l1_loss(org_book_times_scaled, org_b2m_m2b_b)

            # GRADIENT PENALTY
            gradspred_m2b, = th.autograd.grad(org_m1_org_range, org_book_times,
                                       grad_outputs=org_m1_org_range.data.new(org_m1_org_range.shape).fill_(1),
                                       retain_graph=True, create_graph=True)
            gradspred_b2m, = th.autograd.grad(org_b1_org_range, org_movie_times,
                                       grad_outputs=org_b1_org_range.data.new(org_b1_org_range.shape).fill_(1),
                                       retain_graph=True, create_graph=True)

            grad_penalty_m2b = -gradspred_m2b[gradspred_m2b < 0].mean() if len(gradspred_m2b[gradspred_m2b < 0]) > 0 else 0.
            grad_penalty_b2m = -gradspred_b2m[gradspred_b2m < 0].mean() if len(gradspred_b2m[gradspred_b2m < 0]) > 0 else 0.


            # GROUND TRUTH REGRESSION
            if args.loss_scaled == 'yes':
                lossGT_m2b = l1_loss(org_m1[gt_dict[0][train]], th.LongTensor(gt_dict[1][train]).to(device)/org_movie_len)
                lossGT_val_m2b = l1_loss(org_m1[gt_dict[0][val]], th.LongTensor(gt_dict[1][val]).to(device)/org_movie_len)
                lossGTD_m2b = l1_loss(org_m1[gt_dict_dialog[0]], th.FloatTensor(gt_dict_dialog[1]).to(device)/org_movie_len)

                lossGT_b2m = l1_loss(org_b1[gt_dict[1][train]], th.LongTensor(gt_dict[0][train]).to(device)/org_book_len)
                lossGT_val_b2m = l1_loss(org_b1[gt_dict[1][val]], th.LongTensor(gt_dict[0][val]).to(device)/org_book_len)
                lossGTD_b2m = l1_loss(org_b1[gt_dict_dialog[1]], th.FloatTensor(gt_dict_dialog[0]).to(device)/org_book_len)
            else:
                lossGT_m2b = l1_loss(org_m1_org_range[gt_dict[0][train]], th.LongTensor(gt_dict[1][train]).to(device))
                lossGT_val_m2b = l1_loss(org_m1_org_range[gt_dict[0][val]], th.LongTensor(gt_dict[1][val]).to(device))
                lossGTD_m2b = l1_loss(org_m1_org_range[gt_dict_dialog[0]], th.FloatTensor(gt_dict_dialog[1]).to(device))

                lossGT_b2m = l1_loss(org_b1_org_range[gt_dict[1][train]], th.LongTensor(gt_dict[0][train]).to(device))
                lossGT_val_b2m = l1_loss(org_b1_org_range[gt_dict[1][val]], th.LongTensor(gt_dict[0][val]).to(device))
                lossGTD_b2m = l1_loss(org_b1_org_range[gt_dict_dialog[1]], th.FloatTensor(gt_dict_dialog[0]).to(device))

            # WARPING AND RECONSTRUCTION
            # coarse scale for training
            pred_book_feats = reverse_mapping(movie_feats, m1_org_range.squeeze(), kernel_type).to(device)
            lossR_m2b = loss_rec(book_feats, pred_book_feats.to(device))
            pred_movie_feats = reverse_mapping(book_feats, b1_org_range.squeeze(), kernel_type).to(device)
            lossR_b2m = loss_rec(movie_feats, pred_movie_feats.to(device))
            # fine scale for logging
            fine_pred_book_feats = reverse_mapping(org_movie_feats, org_m1_org_range.squeeze(), kernel_type).to(device)
            score_fine_scale_m2b = th.mul(fine_pred_book_feats, org_book_feats).sum(0).mean()
            fine_pred_movie_feats = reverse_mapping(org_book_feats, org_b1_org_range.squeeze(), kernel_type).to(device)
            score_fine_scale_b2m = th.mul(fine_pred_movie_feats, org_movie_feats).sum(0).mean()

            gt_sim_score_m2b = th.mul(fine_pred_book_feats[:, gt_dict[0]], org_book_feats[:, gt_dict[0]]).sum(0).mean()
            gt_sim_score_dialog_m2b = th.mul(fine_pred_book_feats[:, gt_dict_dialog[0]],
                                         org_book_feats[:, gt_dict_dialog[0]]).sum(0).mean()
            gt_sim_score_b2m = th.mul(fine_pred_movie_feats[:, gt_dict[1]], org_movie_feats[:, gt_dict[1]]).sum(0).mean()
            gt_sim_score_dialog_b2m = th.mul(fine_pred_movie_feats[:, gt_dict_dialog[1]],
                                         org_movie_feats[:, gt_dict_dialog[1]]).sum(0).mean()

            # Write to wandb
            wandb.log({'epoch': epoch, 'lr': lr,
                       'loss_cycle_consistency_m2b': lossCC_m2b, 'loss_cycle_consistency_b2m': lossCC_b2m,
                       'grad_penalty_m2b': grad_penalty_m2b, 'grad_penalty_b2m': grad_penalty_b2m,
                       'lossGT_m2b': lossGT_m2b, 'lossGT_val_m2b': lossGT_val_m2b, 'lossGTD_m2b': lossGTD_m2b,
                       'lossGT_b2m': lossGT_b2m, 'lossGT_val_b2m': lossGT_val_b2m, 'lossGTD_b2m': lossGTD_b2m,
                       'coarse_reconstruction_m2b': -lossR_m2b, 'coarse_reconstruction_b2m': -lossR_b2m,
                       'fine_reconstruction_m2b': score_fine_scale_m2b, 'fine_reconstruction_b2m': score_fine_scale_b2m,
                       'gt_reconstruction_m2b': gt_sim_score_m2b, 'gt_reconstruction_b2m': gt_sim_score_b2m,
                       })

            # Backpropagate and update losses
            # to do make GT loss [0,1]
            optimizer.zero_grad()
            loss_ = lossCC_m2b + lossCC_b2m + grad_penalty_m2b + grad_penalty_b2m
            loss_ += lossGT_m2b + lossGT_b2m + lossGTD_m2b + lossGTD_b2m
            loss_ += lossR_m2b + lossR_b2m
            # loss_ = args.gt_loss * lossGT + args.rec_loss * lossR + args.gtd_loss * lossGTD + grad_penalty
            loss_.backward(retain_graph=True)
            loss_now = loss_

            # Optimizer step
            optimizer.step()

            # Only every 5 epochs, visualize images and mapping
            if epoch % args.print_every == 0:
                message = f"Epoch {epoch} [m2b/b2m] loss GT: {lossGT_m2b:.2}/{lossGT_b2m:.2} loss R {lossR_m2b:.4}/{lossR_b2m:.4}"
                message += f" loss CC {lossCC_m2b:.3}/{lossCC_b2m:.3}, penalty {grad_penalty_m2b:.2}/{grad_penalty_b2m:.2}"
                print(message)
                if epoch == 0:
                    # Visualize input and output
                    visualize_input(movie_feats.cpu().data.numpy(), book_feats.cpu().data.numpy())

                # Visualize mapping
                get_plot(org_book_times.cpu().detach().numpy(), org_m1_org_range.detach().cpu(), gt_dict,
                         split={'train': train, 'val': val}, gt_dict_dialog=gt_dict_dialog, dir='M2B')
                get_plot(org_movie_times.cpu().detach().numpy(), org_b1_org_range.detach().cpu(), [gt_dict[1], gt_dict[0]],
                         split={'train': train, 'val': val}, gt_dict_dialog=[gt_dict_dialog[1], gt_dict_dialog[0]], dir='B2M')
                plot_diff(movie_feats.cpu().data.numpy(),
                          pred_book_feats.cpu().data.numpy(),
                          book_feats.cpu().data.numpy(), titles=['Input', 'Prediction', 'Output', 'Difference'])
                plot_grad(gradspred_m2b.cpu().detach().numpy(), dir='M2B')
                plot_grad(gradspred_b2m.cpu().detach().numpy(), dir='B2M')

            epoch += 1

    
    end_time = time.time()
    save_path = f"outputs/{args.movie}"
    th.save(model.state_dict(), f"{save_path}/{exp_name}_model.pt")
    # wandb.save('model.h5')
