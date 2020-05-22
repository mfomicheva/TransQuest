import os

import unittest

from examples.common.train import read_data_files
from examples.common.util.data import load_examples

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestData(unittest.TestCase):

    def test_reads_data(self):
        train_tsv = os.path.join(data_dir, 'et-en', 'train.eten.df.short.tsv')
        test_tsv = os.path.join(data_dir, 'et-en', 'dev.eten.df.short.tsv')
        train_df, test_df = read_data_files(train_tsv, test_tsv)
        assert len(train_df) == 7000
        assert len(test_df) == 1000

    def test_loads_examples(self):
        train_tsv = os.path.join(data_dir, 'et-en', 'train.eten.df.short.tsv')
        test_tsv = os.path.join(data_dir, 'et-en', 'dev.eten.df.short.tsv')
        train_df, test_df = read_data_files(train_tsv, test_tsv)
        examples = load_examples(test_df)
        print(len(examples))


if __name__ == '__main__':
    unittest.main()
