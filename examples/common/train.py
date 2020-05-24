import argparse
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.transformers.evaluation import pearson_corr, spearman_corr
from algo.transformers.run_model import QuestModel
from examples.common.util.draw import draw_scatterplot

from examples.common.config.train_config import train_config
from examples.common.config.train_config import MODEL_TYPE
from examples.common.config.train_config import MODEL_NAME
from examples.common.config.train_config import SEED

from examples.common.util.data import read_data_files
from examples.common.util.normalizer import un_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('--inject_features', nargs='+', default=None)
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    train_config.update({
        'output_dir': os.path.join(args.output_dir, 'outputs'),
        'best_model_dir': os.path.join(args.output_dir, 'best_model'),
        'cache_dir': os.path.join(args.output_dir, 'cache_dir')
    })

    train, test = read_data_files(args.train_path, args.test_path, inject_features=args.inject_features)
    if train_config['evaluate_during_training']:
        if train_config['n_fold'] > 1:
            test_preds = np.zeros((len(test), train_config['n_fold']))
            print(test_preds.shape)
            for i in range(train_config['n_fold']):
                print('Training with N folds. Now N is {}'.format(i))
                if os.path.exists(train_config['output_dir']) and os.path.isdir(train_config['output_dir']):
                    shutil.rmtree(train_config['output_dir'])

                model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                   args=train_config)
                train_n, eval_df_n = train_test_split(train, test_size=0.1, random_state=SEED * i)
                print('Created train set with {} examples and eval set with {} examples'.format(len(train), len(eval_df_n)))
                model.train_model(train_n, eval_df=eval_df_n, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                                  mae=mean_absolute_error, model_scores=bool(args.inject_features))
                model = QuestModel(MODEL_TYPE, train_config['best_model_dir'], num_labels=1,
                                   use_cuda=torch.cuda.is_available(), args=train_config)

                result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                            spearman_corr=spearman_corr,
                                                                            mae=mean_absolute_error)
                test_preds[:, i] = model_outputs

            test['predictions'] = test_preds.mean(axis=1)

        else:
            model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=train_config)
            train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
            model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = QuestModel(MODEL_TYPE, train_config['best_model_dir'], num_labels=1,
                               use_cuda=torch.cuda.is_available(), args=train_config)
            result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            test['predictions'] = model_outputs
    else:
        model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(), args=train_config)
        model.train_model(
            train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error,
            inject_features=args.inject_features
        )
        result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        test['predictions'] = model_outputs

    test = un_fit(test, 'labels')
    test = un_fit(test, 'predictions')
    test.to_csv(os.path.join(args.output_dir, 'results.tsv'), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(test, 'labels', 'predictions', os.path.join(args.output_dir, 'results.png'),
                     MODEL_TYPE + ' ' + MODEL_NAME)


if __name__ == '__main__':
    main()
