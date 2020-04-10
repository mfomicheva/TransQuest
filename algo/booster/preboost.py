from tqdm import tqdm
import torch
from ststransformers.algo.transformer_model import STSTransformerModel
import numpy as np

from examples.common.util.download import download_from_google_drive


def prepare_training_file(model_type, model_name, google_drive_file, google_drive_file_id, train_df, config, column_name):
    if google_drive_file:
        download_from_google_drive(google_drive_file_id, model_name)

    model = STSTransformerModel(model_type, model_name, num_labels=1,
                                use_cuda=torch.cuda.is_available(), args=config)

    sentence_list = train_df[column_name].tolist()
    similarity_sentence_list = []
    similarity_list = []

    for i in tqdm(range(len(sentence_list))):
        sentence = sentence_list[i]
        list_2 = sentence_list[:i] + sentence_list[i+1:]
        temp_test = [[sentence, y] for y in list_2]
        preds, model_outputs = model.predict(temp_test)
        max_ind = np.argmax(preds)

        similarity_sentence_list.append(list_2[max_ind])
        similarity_list.append(preds[max_ind])



