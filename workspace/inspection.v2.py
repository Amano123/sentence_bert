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
dataset_file_path = "/home/yo/workspace/speaker_utterance_dateset.v2"

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

    # pandas dataframe
    cls_cos_matrix_df = pd.DataFrame(cls_cos_matrix)
    mean_cos_matrix_df  = pd.DataFrame(mean_cos_matrix)

    # cls token
    cls_match_assembly_speaker_name = []
    cls_match_assembly_utterance = []
    # sentence-bert meaan
    mean_match_assembly_speaker_name = []
    mean_match_assembly_utterance = []

    for digest_index in digest_df.index:
        # print(f"{digest_df.speaker_name[digest_index]} : {digest_df.utterance[digest_index]}")
        cls_cos_score = cls_cos_matrix_df[cls_cos_matrix_df.index == digest_index].T
        # print((cls_cos_score))
        # mean_cos_score = mean_cos_matrix[digest_index]
        mean_cos_score = mean_cos_matrix_df[mean_cos_matrix_df.index == digest_index].T

        filter_name = digest_df.speaker_name[digest_index]
        # print(filter_name)
        filter_df = assembly_df[(assembly_df.speaker_name == filter_name)].reset_index()
        # print(filter_df)
        filter_id = []

        if set(filter_df.label) == set():
            filter_name = "aaaa"
            cls_match_assembly_speaker_name.append(filter_name)
            cls_match_assembly_utterance.append(filter_name)
            mean_match_assembly_speaker_name.append(filter_name)
            mean_match_assembly_utterance.append(filter_name)
            continue

        for label in set(filter_df.label):
            filter_id.append(label)
            filter_id.append(label + 1)
            filter_id.append(label - 1)
            # filter_id.append(label + 2)
            # filter_id.append(label - 2)
            

        test = assembly_df[assembly_df.label.isin(filter_id)].index
        # 発言者の制限をかけない場合
        # test = assembly_df.index
        # print(test)

        # cls_max_index = cls_cos_score.index(max(cls_cos_score))
        cls_max_index = cls_cos_score[cls_cos_score.index.isin(test)].idxmax()
        # print(cls_max_index)
        # mean_max_index = mean_cos_score.index(max(mean_cos_score))
        mean_max_index = mean_cos_score[mean_cos_score.index.isin(test)].idxmax()
        # print(mean_max_index)

        cls_match_assembly_speaker_name.append(assembly_df.speaker_name[cls_max_index].to_string(index=False))
        # print(assembly_df.speaker_name[cls_max_index].to_string(index=False))
        cls_match_assembly_utterance.append(assembly_df.utterance[cls_max_index].to_string(index=False))
        mean_match_assembly_speaker_name.append(assembly_df.speaker_name[mean_max_index].to_string(index=False))
        mean_match_assembly_utterance.append(assembly_df.utterance[mean_max_index].to_string(index=False))


        # break
        # print()
    # break

    digest_df["cls_match_assembly_speaker_name"] = cls_match_assembly_speaker_name
    digest_df["cls_match_assembly_utterance"] = cls_match_assembly_utterance
    digest_df["mean_match_assembly_speaker_name"] = mean_match_assembly_speaker_name
    digest_df["mean_match_assembly_utterance"] = mean_match_assembly_utterance
    # break

    cls_result = len(digest_df[
        (digest_df.cls_match_assembly_speaker_name == digest_df.speaker_name) & 
        (digest_df.label == "speaker.summury")
        ]) / len(digest_df[(digest_df.label == "speaker.summury")])

    mean_result = len(digest_df[
        (digest_df.mean_match_assembly_speaker_name == digest_df.speaker_name) & 
        (digest_df.label == "speaker.summury")
        ]) / len(digest_df[(digest_df.label == "speaker.summury")])

    del digest_df['cls_embedding']
    del digest_df['mean_embedding']
    digest_df.to_csv(f"/home/yo/workspace/result/{f_name}.csv")
    print(f"{f_name:20}: cls: [{cls_result:.5%}], mean: [{mean_result:.5%}]")


 # %%
