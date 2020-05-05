import logging

import scipy
import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

import pandas as pd
from tqdm import tqdm

from transquest.util.split import batch


def semantic_augmentation(sentence_encoder, files, nmt_training_file, column_name, other_column_name, nmt_column_name, nmt_other_column_name, augment_threshhold):
    embedder = SentenceTransformer(sentence_encoder)

    nmt_sentence_list = nmt_training_file[nmt_column_name].tolist()
    nmt_other_sentence_list = nmt_training_file[nmt_other_column_name].tolist()
    # nmt_sentence_embeddings = []
    # for x in tqdm(batch(nmt_sentence_list, 1024), total=int(len(nmt_sentence_list)/1024)):
    #     nmt_sentence_embeddings.extend(embedder.encode(x, batch_size=1024, show_progress_bar=True))

    nmt_sentence_embeddings = embedder.encode(nmt_sentence_list, batch_size=1024, show_progress_bar=True)

    augmented_files = []
    for file in files:
        sentence_list = file[column_name].tolist()
        sentence_embeddings = []
        for x in tqdm(batch(sentence_list, 1024), total=int(len(sentence_list)/1024)):
            sentence_embeddings.extend(embedder.encode(x, batch_size=1024, show_progress_bar=True))
        similar_sentence_list = []
        other_sentence_list = []
        quality_list = []

        for sentence, sentence_embedding in tqdm(zip(sentence_list, sentence_embeddings), total=len(sentence_list)):
            for nmt_sentence, nmt_sentence_embedding, nmt_other_sentence in zip(nmt_sentence_list, nmt_sentence_embeddings, nmt_other_sentence_list):
                # distances = scipy.spatial.distance.cdist([sentence_embedding], nmt_sentence_embeddings, "cosine")[0]
                #
                # results = zip(range(len(distances)), distances)
                # results = sorted(results, key=lambda x: x[1])
                #
                # idx, distance = results[0]
                # similrity = 1 - distance

                similarity = dot(sentence_embedding, nmt_sentence_embedding) / (norm(sentence_embedding) * norm(nmt_sentence_embedding))

                if similarity > augment_threshhold:
                    similar_sentence_list.append(nmt_sentence)
                    other_sentence_list.append(nmt_other_sentence)
                    quality_list.append(1.0)

        augmented_df = pd.DataFrame(
            {column_name: similar_sentence_list,
            other_column_name : other_sentence_list,
            'labels': quality_list
            })
        augmented_files.append(augmented_df)

    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return augmented_files


def normal_augmentation(nmt_training_file, threshhold):

    cut_nmt_training_file = nmt_training_file.head(1000)
    cut_nmt_training_file["labels"] = 1.0

    return cut_nmt_training_file




