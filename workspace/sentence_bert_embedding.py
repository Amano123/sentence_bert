#%%
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from pandas import DataFrame
import math
import os
from tqdm import tqdm
#%%

CLS_model_path = "/home/yo/workspace/output/CLS_training_nli_cl-tohoku/bert-base-japanese-whole-word-masking_2022-09-02_07-08-17/checkpoint/50000"
MAX_model_path = "/home/yo/workspace/output/max_training_nli_cl-tohoku/bert-base-japanese-whole-word-masking_2022-09-02_07-14-54/checkpoint/50000"
MEAN_model_paht = "/home/yo/workspace/output/mean_training_nli_cl-tohoku/bert-base-japanese-whole-word-masking_2022-09-01_11-00-05/checkpoint/50000"

# model_file_path = CLS_model_path
cls_model = SentenceTransformer(CLS_model_path)
mean_model = SentenceTransformer(MEAN_model_paht)

#%%
dataset_file_path = "/home/yo/workspace/speaker_utterance_dateset"

#%%
def cls_model_embedding(df: DataFrame) -> DataFrame :
    utterance = df.utterance
    embedding = cls_model.encode(utterance, convert_to_tensor=True).cpu()# type: ignore
    df["cls_embedding"] = embedding.tolist()
    return df

def mean_model_embedding(df: DataFrame) -> DataFrame :
    utterance = df.utterance
    embedding = mean_model.encode(utterance, convert_to_tensor=True).cpu()# type: ignore
    df["mean_embedding"] = embedding.tolist()
    return df

# %%
for f_name in tqdm(os.listdir(dataset_file_path), total=len(os.listdir(dataset_file_path))):
    # 都議会だより
    digest_df = pd.read_csv(f"{dataset_file_path}/{f_name}/digest.csv", index_col=0)
    # 都議会会議録
    assembly_df = pd.read_csv(f"{dataset_file_path}/{f_name}/assembly.csv", index_col=0)
    # cls embedding 
    add_cls_model_digest_df = cls_model_embedding(digest_df)
    add_cls_model_assembly_df = cls_model_embedding(assembly_df)
    # mean embedding
    embedded_digest_df = mean_model_embedding(add_cls_model_digest_df)
    embedded_assembly_df = mean_model_embedding(add_cls_model_assembly_df)

    embedded_digest_df.to_pickle(f"{dataset_file_path}/{f_name}/embedded_digest.pkl")
    embedded_assembly_df.to_pickle(f"{dataset_file_path}/{f_name}/embedded_assembly.pkl")
# %%
