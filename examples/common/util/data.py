import pandas as pd

from examples.common.util.normalizer import fit, un_fit
from algo.transformers.utils import InputExample


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
    if "model_scores" in df.columns:
        model_scores = df["model_scores"].to_list()
        for i, ex in enumerate(examples):
            ex.model_score = model_scores[i]
    return examples


def read_data_files(train_file, test_file, inject_features=None):
    train = pd.read_csv(train_file, sep='\t', quoting=3)
    test = pd.read_csv(test_file, sep='\t', quoting=3)

    select_columns = ['original', 'translation', 'z_mean']
    if inject_features is not None:
        select_columns.extend(inject_features)
    train = train[select_columns]
    test = test[select_columns]

    train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
    test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})

    train = fit(train, 'labels')
    test = fit(test, 'labels')
    return train, test
