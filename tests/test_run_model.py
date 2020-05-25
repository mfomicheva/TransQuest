import os
import unittest

from examples.common.train import train_model
from examples.common.train import load_config
from examples.common.util.data import read_data_files

from tests.utils import Args


test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestTrain(unittest.TestCase):

    out_dir = os.path.join(data_dir, 'toy', 'output')
    train_path = os.path.join(data_dir, 'toy', 'toy.tsv')
    config_path = os.path.join(data_dir, 'toy', 'toy.json')
    features_pref = os.path.join(data_dir, 'toy', 'features')
    test_path = train_path
    args = Args(config_path, out_dir)

    def test_trains_model(self):
        config = load_config(self.args)
        config['MODEL_TYPE'] = 'xlmroberta'
        train, test = read_data_files(self.train_path, self.test_path)
        train_model(train, config, test_size=0.5)

    def test_trains_model_with_injected_features(self):
        config = load_config(self.args)
        config['MODEL_TYPE'] = 'xlmrobertainject'
        train, test = read_data_files(self.train_path, self.test_path, features_pref=self.features_pref)
        train_model(train, config, test_size=0.5)
