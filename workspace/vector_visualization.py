#%%
import pandas as pd
import numpy as np
from operator import itemgetter
import torch
#%%
# df = pd.read_pickle("/home/yo/workspace/話者毎に色分け-最終盤/2019-02-20/2019-02-20.pkl")
df = pd.read_pickle("/home/yo/workspace/話者毎に色分け-最終盤/2019-02-25/2019-02-25.pkl")
# df = pd.read_pickle("/home/yo/workspace/話者毎に色分け-最終盤/2020-03-12/2020-03-12.pkl")

# %%
sentence_embeddings = df["sentence_embeddings"][0]
cls_token_embeddings = df["cls_token_embeddings"][0]
#%%
speaker = df["speaker"][0]
# %%
# np.var(np.array(sentence_embeddings[0])),np.var(np.array(cls_token_embeddings[0]))
# %%
spe_index = {}
for index, spe in enumerate(set(speaker)):
    result_index = [i for i, x in enumerate(speaker) if x == spe]
    spe_index[spe] = result_index
# %%
for spe in set(speaker):
    filter_index = spe_index[spe]

    filter_sentence_emb = itemgetter(filter_index)(sentence_embeddings)
    filter_cls_emb = itemgetter(filter_index)(torch.Tensor(cls_token_embeddings))

    distributed_sentence_emb = np.var(np.array(filter_sentence_emb))
    distributed_cls_emb = np.var(np.array(filter_cls_emb))
    print(
        f"{spe}, \n\tsentence distributed: {distributed_sentence_emb}, \n\tCLS distributed     : {distributed_cls_emb}"
        )
# %%
