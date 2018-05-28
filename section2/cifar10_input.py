"""
File: section3/cifar10_input.py
Author: Brandon McKinzie
Description: contains the code for checking whether the user has the CIFAR-10
             dataset, and downloading it for them if it is not found.
"""

import os
import sys
import urllib.request
import tarfile

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract(data_dir='data'):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    os.makedirs(dest_directory, exist_ok=True)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:.2%}%'.format(
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
