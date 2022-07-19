#%%
import pandas as pd
import numpy as np
from operator import itemgetter
import torch
import itertools
import japanize_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections
#%%
# df = pd.read_pickle("/home/yo/workspace/話者毎に色分け-最終盤/2019-02-20/2019-02-20.pkl")
# df = pd.read_pickle("/home/yo/workspace/話者毎に色付け/2019-02-25/2019-02-25.pkl")
date_str = "2019-02-20"
df = pd.read_pickle(f"/home/yo/workspace/話者毎に色付け-2022-07-07/{date_str}/{date_str}.pkl")

# %%
cls_tsne_emnedding_df = pd.DataFrame(df["cls_tsne_emnedding"][0])
sentence_tsne_emnedding_df = pd.DataFrame(df["sentence_tsne_emnedding"][0])

sentence_embeddings_df = pd.DataFrame(df["sentence_embeddings"][0])
cls_token_embeddings_df = pd.DataFrame(df["cls_token_embeddings"][0])
utterance = df["utterance"][0]

#%%
topic_df = pd.read_csv(f"/home/yo/workspace/{date_str}-document_topic.csv")
topic_df["document_topic"] = topic_df.iloc[:, 1:].idxmax(axis=1)

#%%
# クラスタリング　ラベル
sentence_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(sentence_embeddings_df)
sentence_cluster_labels = sentence_kmeans_model.labels_

cls_token_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(cls_token_embeddings_df)
cls_token_cluster_labels = cls_token_kmeans_model.labels_
#%%
# 不要な文章を制限
c = collections.Counter(utterance)
black_list = []
for i in c:
    if 1 != c[i]:
        black_list.append(i)

# %%
## topic fileter
utterance_topic_list = []
for (index, score), utte in zip(enumerate(topic_df.document_topic), utterance):

    if topic_df[score][index] > 0.0:
        if utte in black_list:
            utterance_topic_list.append(False)
        else :
            utterance_topic_list.append(True)
    else:
        utterance_topic_list.append(False)

# %%
cls_tsne_emnedding_df["topic"] = topic_df["document_topic"]
# cls_tsne_emnedding_df["topic_filter"] = utterance_topic_list
cls_tsne_emnedding_df["cluster_label"] = cls_token_cluster_labels

sentence_tsne_emnedding_df["topic"] = topic_df["document_topic"]
# sentence_tsne_emnedding_df["topic_filter"] = utterance_topic_list
sentence_tsne_emnedding_df["cluster_label"] = sentence_cluster_labels

# %%
def plot_function(enbedding_df, label):
    _df = enbedding_df#[enbedding_df["topic_filter"] == True]
    topic_num = len(set(enbedding_df["topic"]))
    # topic_tag = [f"topic{i}" for i in range(1,topic_num +1)]
    topic_tag = set(enbedding_df["cluster_label"])

    fig, ax = plt.subplots(figsize=(8, 8))
    for tag in topic_tag:
        # xy_pickup = _df[_df["topic"] == tag]
        xy_pickup = _df[_df["cluster_label"] == tag]
        x = xy_pickup[0]
        y = xy_pickup[1]
        ax.scatter(x, y, s=6, label=tag)   # type: ignore
        ax.legend(loc='center left', bbox_to_anchor=(1., .5))
    plt.savefig(f'{date_str}-{label}.png', bbox_inches='tight')    


# %%
plot_function(cls_tsne_emnedding_df, "CLS")
plot_function(sentence_tsne_emnedding_df, "sentence")
# %%
