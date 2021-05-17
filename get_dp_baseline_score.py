import json
import numpy as np
import argparse
import ipdb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_score(i, j):
    return sim[i, j]


def getResults(seq1, seq2):
    score, F, TB = seqalignDP(seq1, seq2)
    s1, s2 = traceback(seq1, seq2, TB)
    return score, F, TB, s1, s2


gap_pen = 1
PTR_NONE, PTR_GAP1, PTR_GAP2, PTR_BASE = 0, 1, 2, 3


def traceback(seq1, seq2, TB):
    s1, s2 = [], []
    i, j = len(seq1), len(seq2)
    while TB[i][j] != PTR_NONE:
        if TB[i][j] == PTR_BASE:
            s1.append(seq1[i - 1])
            s2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif TB[i][j] == PTR_GAP1:
            s1.append([])
            s2.append(seq2[j - 1])
            j -= 1
        elif TB[i][j] == PTR_GAP2:
            s1.append(seq1[i - 1])
            s2.append([])
            i -= 1
        else:
            assert False
    return s1, s2


get_set = lambda x: set(x) if len(x) > 1 else x


def seqalignDP(seq1, seq2):
    """return the score of the optimal Needleman-Wunsch alignment for seq1 and seq2
    Note: gap_pen should be positive (it is subtracted)
    """
    F = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    TB = [[PTR_NONE for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

    # initialize dynamic programming table for Needleman-Wunsch alignment (Durbin p.20)
    for i in range(1, len(seq1) + 1):
        F[i][0] = 0 - i * gap_pen
        TB[i][0] = PTR_GAP2  # indicates a gap in seq2
    for j in range(1, len(seq2) + 1):
        F[0][j] = 0 - j * gap_pen
        TB[0][j] = PTR_GAP1  # indicates a gap in seq1

    for i in tqdm(range(1, len(seq1) + 1)):
        for j in range(1, len(seq2) + 1):
            insertGapInSeq2 = F[i - 1][j] - gap_pen
            insertGapInSeq1 = F[i][j - 1] - gap_pen
            matchOrMutation = F[i - 1][j - 1] + get_score(seq1[i - 1], seq2[j - 1])
            max_score = max(insertGapInSeq1, insertGapInSeq2, matchOrMutation)
            #                 print(i, j, insertGapInSeq1, insertGapInSeq2, matchOrMutation)
            F[i][j] = max_score
            if max_score == insertGapInSeq1:
                TB[i][j] = PTR_GAP1
            elif max_score == insertGapInSeq2:
                TB[i][j] = PTR_GAP2
            elif max_score == matchOrMutation:
                TB[i][j] = PTR_BASE
    return F[len(seq1)][len(seq2)], F, TB


def visualize_alignments(s1, s2, start=0, end=100):
    for i, j, _ in zip(s1, s2, range(start, end)):
        print(i, '\n', j, _)
        print('-------')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, default='Harry.Potter.and.the.Sorcerers.Stone', help='movie name')
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

    # assign high score to gt data
    sim[gt_dict[1][train], gt_dict[0][train]] = 1.
    gap_pen = 1
    score, F, TB, s1, s2 = getResults(np.arange(movie_len), np.arange(book_len))
    points = []
    for i,j in zip(s1, s2):
        if type(j) != list:
            points.append((i, j))
    alignment = np.array(points)
    # plt.plot(np.array(points)[:, 0], np.array(points)[:, 1])
    # plt.show()
    np.save(f"data/{args.movie}/dp_alignment.npy", alignment)

