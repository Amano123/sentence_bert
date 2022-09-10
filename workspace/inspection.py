#%%
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from pandas import DataFrame
import math
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

#%%
dataset_file_path = "/home/yo/workspace/speaker_utterance_dateset"

#%%
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# %%
for f_name in os.listdir(dataset_file_path): #tqdm(os.listdir(dataset_file_path), total=len(os.listdir(dataset_file_path))):
    # 都議会だより(embeddig済み)
    digest_df = pd.read_pickle(f"{dataset_file_path}/{f_name}/embedded_digest.pkl")
    # 都議会会議録(embeddig済み)
    assembly_df = pd.read_pickle(f"{dataset_file_path}/{f_name}/embedded_assembly.pkl")
    digest_cls_embedding = digest_df.cls_embedding.tolist()
    assembly_cls_embedding = assembly_df.cls_embedding.tolist()

    digest_mean_embedding = digest_df.mean_embedding.tolist()
    assembly_mean_embedding = assembly_df.mean_embedding.tolist()

    cls_cos_matrix = cosine_similarity(digest_cls_embedding,assembly_cls_embedding)
    mean_cos_matrix = cosine_similarity(digest_mean_embedding,assembly_mean_embedding)

    cls_match_assembly_speaker_name = []
    cls_match_assembly_utterance = []
    mean_match_assembly_speaker_name = []
    mean_match_assembly_utterance = []
    for digest_index in digest_df.index:
        # print(f"{digest_df.speaker_name[digest_index]} : {digest_df.utterance[digest_index]}")
        cls_cos_score = cls_cos_matrix[digest_index].tolist()
        mean_cos_score = mean_cos_matrix[digest_index].tolist()

        cls_max_index = cls_cos_score.index(max(cls_cos_score))
        mean_max_index = mean_cos_score.index(max(mean_cos_score))

        cls_match_assembly_speaker_name.append(assembly_df.speaker_name[cls_max_index])
        cls_match_assembly_utterance.append(assembly_df.utterance[cls_max_index])
        mean_match_assembly_speaker_name.append(assembly_df.speaker_name[mean_max_index])
        mean_match_assembly_utterance.append(assembly_df.utterance[mean_max_index])

    digest_df["cls_match_assembly_speaker_name"] = cls_match_assembly_speaker_name
    digest_df["cls_match_assembly_utterance"] = cls_match_assembly_utterance
    digest_df["mean_match_assembly_speaker_name"] = mean_match_assembly_speaker_name
    digest_df["mean_match_assembly_utterance"] = mean_match_assembly_utterance

    cls_result = len(digest_df[digest_df.cls_match_assembly_speaker_name == digest_df.speaker_name]) / len(digest_df)
    mean_result = len(digest_df[digest_df.mean_match_assembly_speaker_name == digest_df.speaker_name]) / len(digest_df)

    del digest_df['cls_embedding']
    del digest_df['mean_embedding']
    digest_df.to_csv(f"/home/yo/workspace/result/{f_name}.csv")
    print(f"{f_name:20}: cls: [{cls_result:.5%}], mean: [{mean_result:.5%}]")

# %%
