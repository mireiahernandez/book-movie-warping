import json
import numpy as np
import argparse
import ipdb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from network.mlp import MLP, positional_encoding
import torch as th


def get_model_predictions(path, len_output, len_input, pos_encoding):
    model = MLP(1 + 2 * 6, device='cpu:0')
    model = model.to('cpu:0')
    x = th.load('{}'.format(path), map_location=th.device('cpu'))  # test_d2_model.pt')
    model.load_state_dict(x)
    # if not rev:
    #     len_output = book_len
    #     len_input = movie_len
    # else:
    #     len_input, len_output = book_len, movie_len
    output_times = th.FloatTensor(np.arange(len_output)).to('cpu:0')
    output_times_scaled = output_times / (len_output - 1)
    inp1 = positional_encoding(output_times_scaled.unsqueeze(1), num_encoding_functions=pos_encoding)
    invf_output = model.forward(inp1)
    pred_invf_times = invf_output * (len_input - 1)

    gradspred, = th.autograd.grad(pred_invf_times, output_times,
                                  grad_outputs=output_times.data.new(output_times.shape).fill_(1),
                                  retain_graph=True, create_graph=True)

    return pred_invf_times.detach().data.numpy(), gradspred.detach().data.numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, default='Harry.Potter.and.the.Sorcerers.Stone', help='movie name')
    parser.add_argument('--pathNNPEGP', type=str)
    parser.add_argument('--pathNNPE', type=str)

    args = parser.parse_args()
    gt_dict = np.load(f"data/{args.movie}/gt_mapping.npy", allow_pickle=True)
    text_feats, image_feats = np.load(f'data/{args.movie}/text_features.npy'), np.load(f'data/{args.movie}/image_features.npy')
    image_feats = np.asarray(image_feats.T, dtype=np.float64)
    text_feats = np.asarray(text_feats.T, dtype=np.float64)
    text_feats /= np.linalg.norm(text_feats, axis=0, keepdims=True)
    image_feats /= np.linalg.norm(image_feats, axis=0, keepdims=True)
    book_len, movie_len = text_feats.shape[1], image_feats.shape[1]
    sim = image_feats.T @ text_feats
    rng = np.random.default_rng(2021)
    train = sorted(rng.choice(range(len(gt_dict)), len(gt_dict)//2+1, False))
    val = [i for i in range(len(gt_dict)) if i not in train]
    gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]

    predictionsNNPE, gradNNPE = get_model_predictions(args.pathNNPE, book_len, movie_len, 6)
    predictionsNNPER, gradNNPEGP = get_model_predictions(args.pathNNPEGP, book_len, movie_len, 6)

    plt.title('Time function prediction for movie: ', args.movie.replace('.', ' '))
    plt.plot(np.arange(book_len)/book_len, predictionsNNPE/movie_len, label='NN + PE', linewidth=1)
    plt.plot(np.arange(book_len)/book_len, predictionsNNPER/movie_len, label='NN + PE + GP', linewidth=1)

    plt.scatter(gt_dict[0][train]/book_len, gt_dict[1][train]/movie_len, c='b', label='training points', s=3)
    plt.scatter(gt_dict[0][val]/book_len, gt_dict[1][val]/movie_len, c='r', label='validation points', s=3)
    plt.legend()
    plt.xlabel('Time in a Book')
    plt.ylabel('Time in a Movie')
    plt.plot()
    plt.savefig(f"data/{args.movie}/prediction_grad.jpg")

    plt.title('Gradient of the function prediction for movie: ', args.movie.replace('.', ' '))
    plt.plot(gradNNPE, label='NN + PE', linewidth=1)
    plt.plot(gradNNPEGP, label='NN + PE + GP', linewidth=1)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot()
    plt.savefig(f"data/{args.movie}/grad_grad.jpg")



