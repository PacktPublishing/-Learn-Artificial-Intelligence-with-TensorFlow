"""
File: section3/util/glove.py
Author: Brandon McKinzie
"""

import os
import glob
import numpy as np


def get_glove_path(glove_dir, dim=25, prompt_if_multiple_found=True):
    matches = glob.glob('{}/glove.*{}d.txt'.format(glove_dir.rstrip('/'), dim))
    if len(matches) == 0:
        raise FileNotFoundError('Could not find GloVe file for dimension {}.'.format(dim))
    elif len(matches) == 1 or not prompt_if_multiple_found:
        return matches[0]
    else:
        relative_matches = list(map(lambda m: m[m.index(glove_dir):], matches))
        print('\nMultiple GloVe files found with dim={}. '
              'Enter number of choice:\n{}'.format(
                dim, '\n'.join(list(map(lambda i: str(i).replace(',', ':'),
                                    enumerate(relative_matches))))))
        choice = int(input('Number (default=0): ') or 0)
        print('Using: {}\n'.format(os.path.basename(matches[choice])))
        return matches[choice]


def get_glove(dim=25, vocab_size=None, lang=None, prompt_if_multiple_found=True):
    """Load glove word2vec dictionary with vector of size `dim`.
    Args:
        dim: (int) dimensionality of word vectors.
        vocab_size: (int) Number of vectors to get. Default is to get
            all of them in the provided file.
        lang: (str) language to use, e.g. 'en_US'. Default is ignore language.
        prompt_if_multiple_found: (bool) whether to prompt user if multiple
            GloVe files are found with the specified `dim`. If False, choose
            the first match.
    """

    word2vec = {}
    glove_path = get_glove_path('data/glove', dim, prompt_if_multiple_found)
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            'Could not GloVe file: {}. Please go to {} and '
            'download/unzip "glove.6B.zip" to the "glove" '
            'directory.'.format(glove_path, 'https://nlp.stanford.edu/projects/glove/'))

    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            try:
                word2vec[word] = np.fromstring(vec, sep=' ')
            except Exception:
                print('word:', word)
                print('vec:', vec)
                raise ValueError

            if vocab_size and len(word2vec) >= vocab_size:
                break
    return word2vec

