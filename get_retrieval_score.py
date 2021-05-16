import json
import numpy as np
import argparse
import ipdb
from sklearn.linear_model import LinearRegression

def compute_linear_regression_baseline(x, y, train, val, N):
    model = LinearRegression()
    model.fit(x[train].reshape(-1, 1), y[train])
    y_pred = model.predict(x[val].reshape(-1, 1))
    score = np.abs(y_pred - y[val]).mean()/N
    print(100*score)


def compute_retrieval(x, y, gt_x, gt_y):
    r1 = 0
    dot_product = x.T @ y
    for source, target in zip(gt_x, gt_y):
        match = np.argsort(dot_product[source])[::-1][0]
        r1 += np.abs(target - match)/y.shape[1]
        # ipdb.set_trace()

    score = 100*r1/len(gt_x)
    print(f'{score:.5}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    gt_dict = np.load(f"data/{args.movie}/gt_mapping.npy", allow_pickle=True)
    text_feats, image_feats = np.load(f'data/{args.movie}/text_features.npy'), np.load(f'data/{args.movie}/image_features.npy')
    image_feats = np.asarray(image_feats.T, dtype=np.float64)
    text_feats = np.asarray(text_feats.T, dtype=np.float64)
    text_feats /= np.linalg.norm(text_feats, axis=0, keepdims=True)
    image_feats /= np.linalg.norm(image_feats, axis=0, keepdims=True)
    book_len, movie_len = text_feats.shape[1], image_feats.shape[1]

    rng = np.random.default_rng(2021)
    train = sorted(rng.choice(range(len(gt_dict)), len(gt_dict)//2+1, False))
    val = [i for i in range(len(gt_dict)) if i not in train]
    gt_dict = [np.array([i['book_ind'] for i in gt_dict]), np.array([i['movie_ind'] for i in gt_dict])]
    print('Movie to Book')
    compute_retrieval(image_feats, text_feats, gt_dict[1][val], gt_dict[0][val])
    compute_linear_regression_baseline(gt_dict[0], gt_dict[1], train, val, movie_len)
    print('Book to Movie')
    compute_retrieval(text_feats, image_feats, gt_dict[0][val], gt_dict[1][val])
    compute_linear_regression_baseline(gt_dict[1], gt_dict[0], train, val, book_len)