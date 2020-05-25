import os
import argparse
import torch

from transquest.algo.transformers.run_model import QuestModel
from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from sklearn.metrics import mean_absolute_error

from transquest.util.normalizer import un_fit
from transquest.util.draw import draw_scatterplot

from transquest.util.data import read_data_file
from transquest.util.data import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_file')
    parser.add_argument('-m', '--model_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-c', '--config')
    parser.add_argument('--features_pref', default=None, required=False)
    args = parser.parse_args()

    config = load_config(args)
    model = QuestModel(
        config['MODEL_TYPE'], config['best_model_dir'], num_labels=1, use_cuda=torch.cuda.is_available(), args=config
    )
    test = read_data_file(args.test_file)
    result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr,
                                                                mae=mean_absolute_error)
    test['predictions'] = model_outputs
    test = un_fit(test, 'labels')
    test = un_fit(test, 'predictions')
    out_preds = os.path.join(args.output_dir, 'predictions')
    out_preds = '{}.{}.{}'.format(out_preds, config['MODEL_TYPE'], config['MODEL_NAME'])
    out_scatter = os.path.join(args.output_dir, 'scatter')
    out_scatter = '{}.{}.{}'.format(out_scatter, config['MODEL_TYPE'], config['MODEL_NAME'])
    test.to_csv('{}.tsv'.format(out_preds), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(
        test, 'labels', 'predictions', '{}.png'.format(out_scatter), config['MODEL_TYPE'] + ' ' + config['MODEL_NAME'])


if __name__ == '__main__':
    main()
