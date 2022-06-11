#%%
from ast import Index
from datetime import datetime
import math
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator, LabelAccuracyEvaluator
from sentence_bert import Data_Loader
import torch

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%%
model = SentenceTransformer("/home/yo/workspace/output/training_nli_cl-tohoku/bert-base-japanese-whole-word-masking-2022-06-11_05-14-36")
#%%
poliinfo_reader = Data_Loader.poliinfo_uterance("/home/yo/workspace/poliinfo_utterance_dataset")

#%%
utterance, speaker = poliinfo_reader.get_utterance("train", "2019-02-15.tsv")
#%%
utterance_embedding = model.encode(["1","2"], convert_to_tensor=True)
# %%
score_list = []
for INDEX, w in enumerate(utterance_embedding):
    if INDEX == 0:
        continue
    score = util.pytorch_cos_sim(utterance_embedding[INDEX], utterance_embedding[INDEX-1])
    score_list.append(score.cpu().item())
# %%
top_k=None
if top_k:
    top_k = min(top_k, len(utterance))
else:
    top_k = len(utterance)
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, utterance_embedding)[0]
    result = torch.topk(scores, k=top_k)

    print("======")
    print(f"Query: {query}")
    for score, idx in zip(*result):
        print(f"{score.item():.4f} {utterance[idx]}")
# %%
