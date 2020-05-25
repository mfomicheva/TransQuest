import os
import json
import pandas as pd
from multiprocessing import cpu_count

from transquest.util.normalizer import fit
from transquest.algo.transformers.utils import InputExample


def load_config(args):
    config = json.load(open(args.config))
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    config.update({
        'output_dir': os.path.join(args.output_dir, 'outputs'),
        'best_model_dir': os.path.join(args.output_dir, 'best_model'),
        'cache_dir': os.path.join(args.output_dir, 'cache_dir'),
        'process_count': process_count,
    })
    return config


def load_examples(df):
    if "text_a" in df.columns and "text_b" in df.columns:
        examples = [
            InputExample(i, text_a, text_b, label)
            for i, (text_a, text_b, label) in enumerate(
                zip(df["text_a"], df["text_b"], df["labels"])
            )
        ]
    else:
        raise ValueError(
            "Passed DataFrame is not in the correct format. Please rename your columns to text_a, text_b and labels"
        )
    if "feature1" in df.columns:
        for col in df.columns:
            if col.startswith("feature"):
                values = df[col].to_list()
                for i, ex in enumerate(examples):
                    ex.features_inject[col] = values[i]
    return examples


def read_data_files(train_file, test_file, features_pref=None):
    train = pd.read_csv(train_file, sep='\t', quoting=3)
    test = pd.read_csv(test_file, sep='\t', quoting=3)

    select_columns = ['original', 'translation', 'z_mean']

    train = train[select_columns]
    test = test[select_columns]

    train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
    test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})

    train = fit(train, 'labels')
    test = fit(test, 'labels')

    if features_pref is not None:
        features_train = pd.read_csv(features_pref + '.train.tsv', sep='\t')
        features_test = pd.read_csv(features_pref + '.test.tsv', sep='\t')
        assert len(features_train) == len(train)
        assert len(features_test) == len(test)
        assert list(features_train.columns) == list(features_test.columns)
        for column in features_train.columns:
            train[column] = features_train[column]
            test[column] = features_test[column]

    return train, test
