import argparse
import ipdb
import glob
import numpy as np
import scipy.io
from clip_model.clip_encoder import clip_image_encoder
from clip_model.clip_encoder import clip_text_encoder
import os

def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name




dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
frames_by_number_path = '/data/vision/torralba/datasets/movies/data/frames_by_number'
frames_by_number_path2 = '/data/vision/scratch/jomat/frames_by_number'


movies = ['American.Psycho', 'Brokeback.Mountain', 'Fight.Club', 'Gone.Girl',
          'Harry.Potter.and.the.Sorcerers.Stone', 'No.Country.for.Old.Men', 'One.Flew.Over.the.Cuckoo.Nest',
          'Shawshank.Redemption', 'The.Firm', 'The.Green.Mile', 'The.Road']

movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']

'''
missing frames: americn psycho, fight club, No.Country.for.Old.Men, Shawshank.Redemption, 'The.Green.Mile', 'The.Road', the firm
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    
    # Get book sentences
    book_name = getBookName(args.movie)
    book_file = scipy.io.loadmat(book_name)
    book_sentences = book_file['book']['sentences'].item()['sentence']
    book_sentences = [sent[0][0] for sent in book_sentences]

    # Get movie frames
    if os.path.exists(f"{frames_by_number_path}/{args.movie.replace('.', '_')}/"):
        frames_path = f"{frames_by_number_path}/{args.movie.replace('.', '_')}/*"
        frames_dirs = sorted(glob.glob(frames_path))
        frames_files = []
        for frames_dir in frames_dirs:
            frames_files.extend(sorted(glob.glob(frames_dir + '/*')))
        frames_files = frames_files[::10]
    elif os.path.exists(f"{frames_by_number_path2}/{args.movie.replace('.', '_')}/"):
        frames_path = f"{frames_by_number_path2}/{args.movie.replace('.', '_')}/*.jpg"
        frames_dirs = sorted(glob.glob(frames_path))
    else:
        raise FileNotFoundError
    
    # Get book "image" through CLIP text encoder    
    image_features = clip_image_encoder(frames_files)
    image_features = np.array(image_features)
    np.save(f"data/{args.movie}/image_features.npy", image_features)
    print(f"Image features saved at: data/{args.movie}/image_features.npy")
    
    # Get movie "image" through CLIP image encoder
    text_features = clip_text_encoder(book_sentences)
    text_features = np.array(text_features)
    np.save(f"data/{args.movie}/text_features.npy", text_features)
    print(f"Text features saved at: data/{args.movie}/text_features.npy")
