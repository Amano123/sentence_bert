#%%
from ast import Index
from datetime import datetime
import math
import os
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

import umap
from sklearn.datasets import load_digits
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import time
from sklearn.decomposition import PCA
# %%
model = SentenceTransformer("/home/yo/workspace/output/training_nli_cl-tohoku/bert-base-japanese-whole-word-masking-2022-06-11_05-14-36")
#%%
poliinfo_reader = Data_Loader.poliinfo_uterance("/home/yo/workspace/poliinfo_utterance_dataset")
#%%
files = os.listdir("/home/yo/workspace/poliinfo_utterance_dataset/train")
#%%
def main(utterance_list, date_list):
    utterance_embedding = model.encode(utterance_list, convert_to_tensor=True)
    # 発言ごと
    # label = [i for i in range(len(utterance_embedding))]
    # 発話者ごと
    speaker_label = {sp:index for index, sp in enumerate(set(date_list), 1)}
    label = [int(speaker_label[i]) for i in date_list]
    make_t_sne_graph(utterance_embedding.cpu(), date, label)  # type: ignore
# %%

def utterance_data_read(date):
    utterance, speaker = poliinfo_reader.get_utterance("train", date)
    return utterance

def make_t_sne_graph(emnedding, date, label):
    tsne_model = TSNE(
    init='random', 
    n_components=2, 
    random_state = 0, 
    perplexity = 30, 
    n_iter = 1000, 
    learning_rate="auto"
    )
    tsne = tsne_model.fit_transform(emnedding)
    plt.scatter(
        tsne[:,0],
        tsne[:,1],
        s = 15,
        c=label,
        cmap=cm.tab20,  # type: ignore
        alpha=0.4
        )
    plt.colorbar()
    os.makedirs("日付毎に色付け", exist_ok=True)
    plt.savefig(f'./日付毎に色付け/tsne.pdf')    
    plt.clf()
# %%
utterance_list = []
date_list = []
for date in files:
    utterance = utterance_data_read(date)
    utterance_list.extend(utterance)
    date_list.extend([date]* len(utterance))

main(utterance_list, date_list)


# %%
