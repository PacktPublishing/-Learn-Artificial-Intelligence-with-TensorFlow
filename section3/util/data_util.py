import re
from nltk import word_tokenize
from collections import OrderedDict, Counter

_PAD_TOKEN = '_PAD'
_UNK_TOKEN = '_UNK'
START_VOCAB = OrderedDict([(_PAD_TOKEN, 0), (_UNK_TOKEN, 1)])


def newsgroups_line_generator(file_path):
    with open(file_path) as newsgroups_file:
        for line in newsgroups_file:
            label, text = re.split(r'\s', line, 1)
            if not text:
                print('Skipping bad line:', line)
                continue
            yield label, text


def amazon_line_generator(file_path):
    with open(file_path) as amazon_file:
        for line in amazon_file:
            label, text = re.split(r'\s', line, 1)
            text = text.strip("\"")
            if not text:
                print('Skipping bad line:', line)
                continue
            yield label, text


def fit_and_extract(file_path, vocab_size):

    if 'amazon' in file_path:
        line_generator = amazon_line_generator(file_path)
    elif 'newsgroups' in file_path:
        line_generator = newsgroups_line_generator(file_path)
    else:
        raise RuntimeError('Unknown data source: {}'.format(file_path))

    labels = []
    tokenized_texts = []
    word_counter = Counter()
    for i, (label, text) in enumerate(line_generator):
        if i % 100 == 0:
            print('\rProcessing line {}...'.format(i), end='', flush=True)

        tokens = list(map(str.lower, word_tokenize(text)))
        for token in tokens:
            word_counter[token] += 1

        labels.append(label)
        tokenized_texts.append(tokens)

    vocab = [w for w, _ in word_counter.most_common(vocab_size - len(START_VOCAB))]
    vocab = list(START_VOCAB.keys()) + vocab
    return vocab, labels, tokenized_texts


