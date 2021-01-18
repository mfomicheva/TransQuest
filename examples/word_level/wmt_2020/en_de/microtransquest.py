from sklearn.model_selection import train_test_split
import os
from examples.word_level.common.util import reader
from examples.word_level.wmt_2020.en_de.microtransquest_config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEST_PATH, TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE, TEST_TARGET_TAGS_FLE, SEED
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
from transquest.algo.word_level.microtransquest.format import prepare_data, prepare_testdata, post_process
import pandas as pd


if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

raw_train_df = reader(TRAIN_PATH, microtransquest_config, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE, TRAIN_TARGET_TAGS_FLE)
raw_test_df = reader(TEST_PATH, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE)

# raw_train_df.to_csv("train.csv", sep='\t')

test_sentences = prepare_testdata(raw_test_df, args=microtransquest_config)

fold_sources_tags = []
fold_targets_tags = []

for i in range(microtransquest_config["n_fold"]):

    if microtransquest_config["evaluate_during_training"]:
        raw_train, raw_eval = train_test_split(raw_train_df, test_size=0.1, random_state=SEED * i)
        train_df = prepare_data(raw_train, args=microtransquest_config)
        eval_df = prepare_data(raw_eval, args=microtransquest_config)
        tags = train_df['labels'].unique().tolist()
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)
        model.train_model(train_df, eval_df=eval_df)
        model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=tags,
                                     args=microtransquest_config)

    else:
        train_df = prepare_data(raw_train_df, args=microtransquest_config)
        tags = train_df['labels'].unique().tolist()
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)
        model.train_model(train_df)

    predicted_labels, raw_predictions = model.predict(test_sentences, split_on_space=True)
    sources_tags, targets_tags = post_process(predicted_labels, test_sentences, args=microtransquest_config)
    fold_sources_tags.append(sources_tags)
    fold_targets_tags.append(targets_tags)

fold_sources_tags_df = pd.DataFrame.from_records(fold_sources_tags)
fold_targets_tags_df = pd.DataFrame.from_records(fold_targets_tags)

fold_sources_tags_df.to_csv("source.csv", sep='\t')
fold_targets_tags_df.to_csv("source.csv", sep='\t')







# model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)
#
# if microtransquest_config["evaluate_during_training"]:
#     train_df, eval_df = train_test_split(train_df, test_size=0.1,  shuffle=False)
#     model.train_model(train_df, eval_df=eval_df)
#
# else:
#     model.train_model(train_df)
#
# # model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=tags, args=microtransquest_config)
# # test_sentences = prepare_testdata(test_source_sentences, test_target_sentences, args=microtransquest_config)
#
# predicted_labels, raw_predictions = model.predict(test_sentences, split_on_space=True)
#
# sources_tags, targets_tags = post_process(predicted_labels, test_sentences, args=microtransquest_config)

# with open(os.path.join(TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w') as f:
#     for _list in sources_tags:
#         for _string in _list:
#             f.write(str(_string) + ' ')
#         f.write(str('\n'))
#
#
# with open(os.path.join(TEMP_DIRECTORY, TEST_TARGET_TAGS_FLE), 'w') as f:
#     for _list in targets_tags:
#         for _string in _list:
#             f.write(str(_string) + ' ')
#         f.write(str('\n'))








