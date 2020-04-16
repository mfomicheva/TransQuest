import progressbar
import urllib.request
import tarfile
import os

import pandas as pd

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def read_nmt_trainingfile(url, file_name, path, source, target):
    urllib.request.urlretrieve(url, file_name, show_progress)

    tar = tarfile.open(file_name, "r:gz")
    tar.extractall(path)
    tar.close()

    with open(os.path.join(path,source)) as f:
        source_lines = f.readlines()

    with open(os.path.join(path,target)) as f:
        target_lines = f.readlines()

    nmt_training_file = pd.DataFrame(
        {"text_a": source_lines,
         "text_b": target_lines
         })

    return nmt_training_file

