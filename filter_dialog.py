from nltk import word_tokenize
import numpy as np
import pysrt
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import argparse

get_time_in_seconds = lambda a: 60*60*a.hours+60*a.minutes+a.seconds +0.001*a.milliseconds

def get_book_dialogs(book):
    DIALOG_ACROSS_SENTENCES = False
    dialogs = []
    ids = []
    short_ids = []
    for i, sentence in enumerate(book):
        sentence = sentence.replace('\n', '')
        if sentence.count('"') == 2:
            dialogs.append(sentence)
            ids.append(i)
            short_ids.append(i)
        elif sentence.count('"') % 2 == 1:
            DIALOG_ACROSS_SENTENCES = not DIALOG_ACROSS_SENTENCES
            ids.append(i)
            dialogs.append(sentence)
        elif sentence.count('"') % 2 == 0 and sentence.count('"')>0:
            DIALOG_ACROSS_SENTENCES = not DIALOG_ACROSS_SENTENCES
            ids.append(i)
            dialogs.append(sentence)
    return short_ids, ids, dialogs


def get_dialog_cc(subs_path):
    subs = pysrt.open(subs_path, encoding='iso-8859-1')
    dialog_cc = []
    time_in_movie = []
    for i in range(1, len(subs)):
        if not (subs[i].text.startswith('[') and subs[i].text.endswith(']')):
            sentence = subs[i].text.replace('\n', '')
            dialog_cc.append(sentence)
            start = get_time_in_seconds(subs[i].start)
            end = get_time_in_seconds(subs[i].end)
            time_in_movie.append(int(.5*(start+end))*2)

    lens = [len(word_tokenize(i)) for i in dialog_cc]
    print('Script original: Min: {}, Max: {}, Mean: {}'.format(min(lens), max(lens), np.mean(lens)))
    return dialog_cc, time_in_movie


if __name__ == '__main__':
    '''
    Extracts the features from the book and movie:
    - currently assumes the dialogs in the book are in " quotation marks
    - frames are extracted at 2 fps
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--movie', default='data/Harry.Potter.and.the.Sorcerers.Stone_GT', type=str, help='movie name')
    args = parser.parse_args()

    extract_features = False

    # Read the book (sentences in txt format from the original annotations)
    with open('{}/book.txt'.format(args.movie), 'r') as f:
        book = f.readlines()

    # Get the dialogs based on quotation marks
    short_dialogs, dialog_times_in_book, dialog_book = get_book_dialogs(book)

    # Read the srt closed captions file and get the dialogs
    dialog_cc, time_in_movie = get_dialog_cc('{}/cc.srt'.format(args.movie))

    # Saw raw text for analysis
    np.save('{}/dialog_book_text.npy'.format(args.movie), dialog_book)
    np.save('{}/dialog_movie_text.npy'.format(args.movie), dialog_cc)

    if extract_features:
        # Extract features
        model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        dialog_book_embeddings = model.encode(dialog_book, convert_to_tensor=True)
        dialog_cc_embeddings = model.encode(dialog_cc, convert_to_tensor=True)

        # Save the features and times
        np.save('{}/dialog_book_features.npy'.format(args.movie), dialog_book_embeddings.data.numpy())
        np.save('{}/dialog_cc_features.npy'.format(args.movie), dialog_cc_embeddings.data.numpy())
    np.save('{}/dialog_times_dict.npy'.format(args.movie), {'dialog_book_times': dialog_times_in_book,
                                                            'short_dialogs_book': short_dialogs,
                                                            'dialog_movie_times': time_in_movie})

