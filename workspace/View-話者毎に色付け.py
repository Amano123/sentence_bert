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
import japanize_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import umap
from sklearn.datasets import load_digits
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import time
from sklearn.decomposition import PCA

import pandas as pd

from tqdm import tqdm


from operator import itemgetter
# %%
model = SentenceTransformer("/home/yo/workspace/output/training_nli_cl-tohoku/bert-base-japanese-whole-word-masking-2022-06-11_05-14-36")
#%%
poliinfo_reader = Data_Loader.poliinfo_uterance("/home/yo/workspace/poliinfo_utterance_dataset")
#%%
files = os.listdir("/home/yo/workspace/poliinfo_utterance_dataset/train")
#%%
def main(date):
    utterance, speaker = poliinfo_reader.get_utterance("train", date+".tsv")
    sentence_embeddings = model.encode(utterance, convert_to_tensor=True).cpu()
    token_embeddings = model.encode(utterance, output_value="token_embeddings", convert_to_tensor=True)
    # sentence_embeddings = model.encode(utterance).tolist()  # type: ignore
    cls_token_embeddings = []
    for token_embedding in token_embeddings:
        cls_token_embeddings.append(token_embedding.cpu().tolist()[0])

    # 発言ごと
    # label = [i for i in range(len(utterance_embedding))]
    # 発話者ごと
    speaker_label = {sp:index for index, sp in enumerate(set(speaker), 1)}
    speaker_utterance_index = {}
    for spe in set(speaker):
        speaker_utterance_index[spe] = [i for i, x in enumerate(speaker) if x == spe]
    label = [int(speaker_label[i]) for i in speaker]
    cls_tsne_emnedding = make_t_sne_graph(cls_token_embeddings, "CLS-token", date, label, speaker, speaker_utterance_index)# type: ignore
    sentence_tsne_emnedding = make_t_sne_graph(torch.tensor(sentence_embeddings), "sentence" ,date, label, speaker, speaker_utterance_index)# type: ignore
    pandas_save(date, speaker, utterance, cls_token_embeddings, sentence_embeddings, cls_tsne_emnedding, sentence_tsne_emnedding)

def pandas_save(date, speaker, utterance, cls_token_embeddings, sentence_embeddings, cls_tsne_emnedding, sentence_tsne_emnedding):
    pd.DataFrame(
        {
            "speaker": [speaker], 
            "utterance": [utterance],
            "cls_token_embeddings": [cls_token_embeddings], 
            "sentence_embeddings": [sentence_embeddings], 
            "cls_tsne_emnedding": [cls_tsne_emnedding],
            "sentence_tsne_emnedding": [sentence_tsne_emnedding]

        }
    ).to_pickle(f"話者毎に色付け/{date}/{date}.pkl")
# %%
def make_t_sne_graph(emnedding, data_label: str, date, label, speaker, speaker_utterance_index):
    tsne_model = TSNE(
    init='random', 
    n_components=2, 
    random_state = 0, 
    perplexity = 30, 
    n_iter = 1000, 
    learning_rate="auto"
    )
    tsne_emnedding = tsne_model.fit_transform(np.array(emnedding))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, spe in tqdm(enumerate(set(speaker)), total=len(set(speaker))):
        utterance_index = speaker_utterance_index[spe]
        xy_pickup = itemgetter(utterance_index)(tsne_emnedding)
        (x, y) = list(zip(*xy_pickup))
        ax.scatter(x, y, s=6, label=spe)   # type: ignore
        ax.legend(loc='center left', bbox_to_anchor=(1., .5))

    os.makedirs(f"話者毎に色付け/{date}", exist_ok=True)
    plt.savefig(f'./話者毎に色付け/{date}/{data_label}-tsne.eps', bbox_inches='tight')    
    plt.savefig(f'./話者毎に色付け/{date}/{data_label}-tsne.pdf', bbox_inches='tight')    
    plt.clf()
    return tsne_emnedding

# %%
for date in files:
    main(date.split(".")[0])
    
# %%
