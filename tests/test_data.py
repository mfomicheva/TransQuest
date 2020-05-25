import os

import unittest

from examples.common.train import read_data_files
from examples.common.util.data import load_examples
from algo.transformers.run_model import QuestModel
from examples.common.train import load_config

from tests.utils import Args

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestData(unittest.TestCase):

    train_tsv = os.path.join(data_dir, 'toy', 'toy.tsv')  # TODO: This is common between the tests, extract
    test_tsv = os.path.join(data_dir, 'toy', 'toy.tsv')
    config_path = os.path.join(data_dir, 'toy', 'toy.json')
    features_pref = os.path.join(data_dir, 'toy', 'features')
    out_dir = os.path.join(data_dir, 'toy', 'output')
    args = Args(config_path, out_dir)

    def test_reads_data(self):
        train_df, test_df = read_data_files(self.train_tsv, self.test_tsv)
        assert len(train_df) == 9
        assert len(test_df) == 9

    def test_reads_data_with_injected_features(self):
        train_df, test_df = read_data_files(self.train_tsv, self.test_tsv, features_pref=self.features_pref)
        assert train_df.shape == (9, 5)
        assert test_df.shape == (9, 5)

    def test_loads_examples(self):
        train_df, test_df = read_data_files(self.train_tsv, self.test_tsv)
        examples = load_examples(test_df)
        assert len(examples) == 9

    def test_loads_examples_with_features(self):
        train_df, test_df = read_data_files(self.train_tsv, self.test_tsv, features_pref=self.features_pref)
        examples = load_examples(test_df)
        assert len(examples) == 9
        for ex in examples:
            assert ex.features_inject['feature1'] == 0.2
            assert ex.features_inject['feature2'] == 0.5

    def test_loads_and_caches_examples_with_features(self):
        train_df, test_df = read_data_files(self.train_tsv, self.test_tsv, features_pref=self.features_pref)
        examples = load_examples(test_df)
        config = load_config(self.args)
        m = QuestModel(config['MODEL_TYPE'], config['MODEL_NAME'], args=config, use_cuda=False)
        dataset = m.make_dataset(examples)
        assert len(dataset.tensors) == 5
        assert dataset.tensors[4].shape == (9, 2)


if __name__ == '__main__':
    unittest.main()
