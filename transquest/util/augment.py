import logging

import scipy
import torch
from sentence_transformers import SentenceTransformer

import pandas as pd
from tqdm import tqdm


def augment_file(sentence_encoder, file, nmt_training_file, column_name, other_column_name, nmt_column_name, nmt_other_column_name, augment_threshhold):
    embedder = SentenceTransformer(sentence_encoder)

    sentence_list = file[column_name].tolist()
    nmt_sentence_list = nmt_training_file[nmt_column_name].tolist()
    nmt_other_sentence_list = nmt_training_file[nmt_other_column_name].tolist()

    sentence_embeddings = embedder.encode(sentence_list)
    nmt_sentence_embeddings = embedder.encode(nmt_sentence_list)

    logging.info("Finished getting embeddings")

    del embedder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    similar_sentence_list = []
    other_sentence_list = []
    quality_list = []

    for sentence, sentence_embedding in tqdm(zip(sentence_list, sentence_embeddings), total=len(sentence_list)):
        distances = scipy.spatial.distance.cdist([sentence_embedding], nmt_sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        idx, distance = results[0]
        similrity = 1 - distance

        if similrity > augment_threshhold :
            similar_sentence_list.append(nmt_sentence_list[idx])
            other_sentence_list.append(nmt_other_sentence_list[idx])
            quality_list.append(1.0)

    aumented_df = pd.DataFrame(
        { column_name: similar_sentence_list,
          other_column_name : other_sentence_list,
         'labels': quality_list
         })

    return aumented_df
