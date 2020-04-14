from tqdm import tqdm
import torch
from ststransformers.algo.transformer_model import STSTransformerModel
import numpy as np

from examples.common.util.download import download_from_google_drive


def prepare_training_file(model_type, model_name, google_drive_file, google_drive_file_id, test_df, config, column_name, train_df=None):
    if google_drive_file:
        download_from_google_drive(google_drive_file_id, model_name)

    model = STSTransformerModel(model_type, model_name, num_labels=1,
                                use_cuda=torch.cuda.is_available(), args=config)

    sentence_list = test_df[column_name].tolist()
    similarity_sentence_list = []
    similarity_list = []
    quality_list = []

    for i in tqdm(range(len(sentence_list))):
        sentence = sentence_list[i]
        if train_df is None:
            list_2 = sentence_list[:i] + sentence_list[i+1:]
        else:
            list_2 = train_df[column_name].tolist()
        temp_test = [[sentence, y] for y in list_2]
        preds, model_outputs = model.predict(temp_test)
        max_ind = np.argmax(preds)

        similarity_sentence_list.append(list_2[max_ind])
        similarity_list.append(preds[max_ind])

    for i in tqdm(range(len(similarity_sentence_list))):
        similarity_sentence = similarity_sentence_list[i]
        quality = test_df.loc[test_df[column_name] == similarity_sentence, 'A'].iloc[0]
        quality_list.append(quality)

    test_df['similar_sentence'] = similarity_sentence_list
    test_df['similarity'] = similarity_list
    test_df['similar_sentence_quality'] = quality_list

    return test_df
