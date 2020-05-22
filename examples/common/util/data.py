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
