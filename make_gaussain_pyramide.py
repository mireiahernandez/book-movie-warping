import numpy as np
import json
import argparse
import scipy.signal as signal
import matplotlib.pyplot as plt
from PIL import Image, ImageOps



def get_image_pyramid(input, levels=5, type='real'):
    if type == 'real':
        assert input.shape[0] == 512
    images = []
    kernel = np.array([[1, 4, 6, 4, 1]])/16
    for i in range(levels):
        images.append(input)
        new_input = signal.convolve2d(input, kernel, mode='same')
        new_input = new_input[:, ::2]
        new_input /= np.linalg.norm(new_input, axis=0, keepdims=True)
        input = new_input
    return images


def plot_pyramid(images, out_path):
    rows, cols = images[0].shape
    composite_image = np.ones((rows*len(images), cols), dtype=np.double)

    composite_image[:rows, :cols] = images[0]

    i_row = 1
    for _, p in enumerate(images[1:]):
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row*rows:(i_row+1) * rows, :n_cols] = p
        i_row += 1

    fig, ax = plt.subplots()
    data = ax.imshow(composite_image, cmap='gray')
    fig.colorbar(data, ax=ax)
    plt.savefig(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='Harry.Potter.and.the.Sorcerers.Stone_GT', type=str, help='movie name')
    args = parser.parse_args()

    image_feats = np.load(f"data/{args.movie}/image_features.npy").T
    text_feats = np.load(f"data/{args.movie}/text_features.npy").T
    gt_dict = json.load(open(f"data/{args.movie}/gt_mapping.json", 'r'))
    gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]

    image_feats = np.asarray(image_feats, dtype=np.float64)
    text_feats = np.asarray(text_feats, dtype=np.float64)
    text_feats /= np.linalg.norm(text_feats, axis=0, keepdims=True)
    image_feats /= np.linalg.norm(image_feats, axis=0, keepdims=True)

    text_pyramid = get_image_pyramid(text_feats)
    plot_pyramid(text_pyramid, out_path='text_pyramid.jpg')

    image_pyramid = get_image_pyramid(image_feats)
    plot_pyramid(image_pyramid, out_path='image_pyramid.jpg')

# Check on a real image
# text_feats = np.array(ImageOps.grayscale(Image.open('landscape_example.jpeg')).resize(( 730*2, 365*2))).T
# images = get_image_pyramid(text_feats.T, type='image')
# plot_pyramid(images)


