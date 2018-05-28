"""Wrapper for DataNode to support the kinds of queries done by a client."""

import os
import pickle
import threading

from lib import PATHS, util
from lib.data_tree import DataTree
from lib.preprocessing.read import get_dirpaths
from collections import OrderedDict
from operator import itemgetter


class ClientVectorizer:

    def __init__(self, data_config):
        self._pretrained_vectorizer_path = os.path.join(
            DataTree.get_output_dir(data_config), 'vectorizer.pkl')
        self._vectorizer = pickle.load(open(self._pretrained_vectorizer_path, 'rb'))

    def doc_path_to_matrix(self, doc_path):
        with open(doc_path, encoding='utf-8', errors='ignore') as f:
            doc = f.read().strip().lower()
        return self._vectorizer.docs_to_matrix(
            docs=[doc], _print_info=os.path.basename(doc_path))

    def __getattr__(self, item):
        if hasattr(self._vectorizer, item):
            return getattr(self._vectorizer, item)
        else:
            raise AttributeError('{} does not have attribute {}.'.format(
                self.__class__.__name__, item))


class ClientPathTree:

    def __init__(self, base_name):
        self.config_name = base_name
        self.base_dir = PATHS('models', 'servables')
        self.current_dir = self.base_dir
        self.prediction_paths = dict()

    def update_prediction_path(self, path_name, category, probability):
        if path_name not in self.prediction_paths:
            self.prediction_paths[path_name] = []
        self.prediction_paths[path_name].append((category, probability))

    def is_servable_category(self, category):
        return str(category) in self.categories

    def draw(self, path_name):

        def print_row(category, probability):
            category_width = 30
            probability_width = 5
            print('{c:<{cw}} {p:<{pw}}'.format(
                cw=category_width, pw=probability_width,
                c=category, p=probability))

        print()
        print_row('Category', 'Probability')
        print_row('-' * len('Category'), '-' * len('Probability'))
        for category, probability in self.prediction_paths[path_name]:
            print_row(category, '{:.3%}'.format(probability))

    @property
    def servable_dirs(self):
        return [d for d in get_dirpaths(self.base_dir, nested=False)
                if self.config_name in d]

    @property
    def categories(self):
        dirnames = [os.path.basename(d) for d in self.servable_dirs]
        return [name.rsplit('__', 1)[-1] for name in dirnames]



class ResultCounter(object):
    """Counter for the prediction results, in the case where we want asynchronous
    requests. See model_client.py for example usage."""

    def __init__(self, num_tests, concurrency, batch_size):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()
        self.batch_size = batch_size

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / (self.batch_size * float(self._num_tests))

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1