import json
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

FRAME_NEIGHBOURHOOD = 10
SENTENCE_NEIGHBOURHOOD = 10
dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
frames_by_number_path = '/data/vision/torralba/datasets/movies/data/frames_by_number'


movies = ['American.Psycho', 'Brokeback.Mountain', 'Fight.Club', 'Gone.Girl',
          'Harry.Potter.and.the.Sorcerers.Stone', 'No.Country.for.Old.Men', 'One.Flew.Over.the.Cuckoo.Nest',
          'Shawshank.Redemption', 'The.Firm', 'The.Green.Mile', 'The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    original_gt = json.load(open(os.path.join(bksnmvs_path,
                                              'ground_truth_annotation', f'content_{args.movie}.json')))
    text_feats, image_feats = np.load(f'data/{args.movie}/text_features.npy'), np.load(f'data/{args.movie}/image_features.npy')
    image_feats = np.asarray(image_feats.T, dtype=np.float64)
    text_feats = np.asarray(text_feats.T, dtype=np.float64)
    text_feats /= np.linalg.norm(text_feats, axis=0, keepdims=True)
    image_feats /= np.linalg.norm(image_feats, axis=0, keepdims=True)
    book_len, movie_len = text_feats.shape[1], image_feats.shape[1]

    new_gt = []
    for o in original_gt:
        start_frame_time = [float(i) for i in o['Time Shot'].split(' ')[0].split(':')]
        start_frame_time = 2*(start_frame_time[0]*60**2 + start_frame_time[1]*60 + start_frame_time[2])
        end_frame_time = [float(i) for i in o['Time Shot'].split(' ')[1].split(':')]
        end_frame_time = 2*(end_frame_time[0]*60**2 + end_frame_time[1]*60 + end_frame_time[2])

        frame_time = np.array([i for i in range(int(start_frame_time)-FRAME_NEIGHBOURHOOD, int(end_frame_time)+FRAME_NEIGHBOURHOOD)])
        book_time = np.array([i for i in range(o['id_sentence']-SENTENCE_NEIGHBOURHOOD, min(book_len, int(o['id_sentence'])+SENTENCE_NEIGHBOURHOOD))])
        x = text_feats[:, book_time].T @ image_feats[:, frame_time]
        ind = np.unravel_index(np.argsort(x, axis=None)[::-1], x.shape)
        found, j = False, 0
        while not found:
            if not (book_time[ind[0][j]] in [i['book_ind'] for i in new_gt] or frame_time[ind[1][j]] in [i['movie_ind']
                                                                                                          for i in
                                                                                                          new_gt]):
                b, t = ind[0][j], ind[1][j]
                found = True
            else:
                j += 1
        atype = [int('Dialog' in o['Alignment Type']), int('Visual' in o['Alignment Type']), int('Sound' in o['Alignment Type'])]
        new_gt.append({'book_ind': book_time[b], 'movie_ind': frame_time[t], 'type': atype})

    x = text_feats.T @ image_feats
    print('Min score:', x.min(), 'Max score:', x.max(), 'Mean score:', x.mean())
    sns.displot(x.flatten())
    plt.savefig(f"data/{args.movie}/clip_score_distribution.jpg")
    np.save(f"data/{args.movie}/gt_mapping.npy", new_gt)
    print(f"Ground truth saved at: data/{args.movie}/gt_mapping.npy")
