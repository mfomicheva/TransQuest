import os
import csv
import argparse
import pandas as pd
import torch

from algo.transformers.run_model import QuestModel
from algo.transformers.evaluation import pearson_corr, spearman_corr
from sklearn.metrics import mean_absolute_error

from examples.common.util.normalizer import fit, un_fit
from examples.common.config.predict_config import predict_config
from examples.common.config.predict_config import MODEL_TYPE
from examples.common.config.predict_config import MODEL_NAME
from examples.common.util.draw import draw_scatterplot


def read_test_file(test_file):
    test = pd.read_csv(test_file, sep='\t', quoting=csv.QUOTE_NONE)
    test = test[['original', 'translation', 'z_mean']]
    test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
    test = fit(test, 'labels')
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_file')
    parser.add_argument('-m', '--model_dir')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()

    predict_config.update({'best_model_dir': args.model_dir, 'output_dir': args.output_dir, 'cache_dir': args.output_dir})
    model = QuestModel(MODEL_TYPE, predict_config['best_model_dir'], num_labels=1,
                       use_cuda=torch.cuda.is_available(), args=predict_config)
    test = read_test_file(args.test_file)
    result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr,
                                                                mae=mean_absolute_error)
    test['predictions'] = model_outputs
    test = un_fit(test, 'labels')
    test = un_fit(test, 'predictions')
    out_preds = os.path.join(args.output_dir, 'predictions')
    out_preds = '{}.{}.{}'.format(out_preds, MODEL_TYPE, MODEL_NAME)
    out_scatter = os.path.join(args.output_dir, 'scatter')
    out_scatter = '{}.{}.{}'.format(out_scatter, MODEL_TYPE, MODEL_NAME)
    test.to_csv('{}.tsv'.format(out_preds), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(test, 'labels', 'predictions', '{}.png'.format(out_scatter), MODEL_TYPE + ' ' + MODEL_NAME)


if __name__ == '__main__':
    main()
