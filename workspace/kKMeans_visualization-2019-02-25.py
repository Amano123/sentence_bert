#%%
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

from operator import itemgetter
import torch
import itertools
import japanize_matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections
#%%
## データセット指定
date_str = "2019-02-25"
df = pd.read_pickle(f"/home/yo/workspace/話者毎に色付け-2022-07-07/{date_str}/{date_str}.pkl")

# %%
## T-sneのデータ抽出
sentence_tsne_emnedding_df = pd.DataFrame(df["sentence_tsne_emnedding"][0])
cls_tsne_emnedding_df = pd.DataFrame(df["cls_tsne_emnedding"][0])

# emneddingを抽出
sentence_embeddings_df = pd.DataFrame(df["sentence_embeddings"][0])
cls_token_embeddings_df = pd.DataFrame(df["cls_token_embeddings"][0])
utterance = df["utterance"][0]

#%%
# Topic データを読み込む
topic_df = pd.read_csv(f"/home/yo/workspace/{date_str}-document_topic.csv")
topic_df["document_topic"] = topic_df.iloc[:, 1:].idxmax(axis=1)

#%%
# クラスタリング　ラベル
sentence_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(sentence_embeddings_df)
sentence_tnse_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(sentence_tsne_emnedding_df)
sentence_cluster_labels = sentence_kmeans_model.labels_

cls_token_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(cls_token_embeddings_df)
cls_token_tnse_kmeans_model = KMeans(n_clusters=2, random_state=1).fit(cls_tsne_emnedding_df)
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
sentence_tsne_emnedding_df["topic"] = topic_df["document_topic"]
# sentence_tsne_emnedding_df["topic_filter"] = utterance_topic_list
sentence_tsne_emnedding_df["cluster_label"] = sentence_cluster_labels

cls_tsne_emnedding_df["topic"] = topic_df["document_topic"]
# cls_tsne_emnedding_df["topic_filter"] = utterance_topic_list
cls_tsne_emnedding_df["cluster_label"] = cls_token_cluster_labels


# %%
def plot_function(enbedding_df, label, cluster_centers_):
    _df = enbedding_df#[enbedding_df["topic_filter"] == True]
    topic_num = len(set(enbedding_df["topic"]))
    # topic_tag = [f"topic{i}" for i in range(1,topic_num +1)]
    topic_tag = set(enbedding_df["cluster_label"])

    fig, ax = plt.subplots(figsize=(8, 8))
    # 節のマッピング
    for index, tag in enumerate(topic_tag, 1):
        # xy_pickup = _df[_df["topic"] == tag]
        xy_pickup = _df[_df["cluster_label"] == tag]
        x = xy_pickup[0]
        y = xy_pickup[1]
        ax.scatter(x, y, s=6, label=f"cluster {index}")   # type: ignore
        ax.legend(loc='center left', bbox_to_anchor=(1., .5))
    for index, xy in enumerate(cluster_centers_, 1):
        x = xy[0]
        y = xy[1]
        ax.scatter(x, y, s=50, marker="x", c="red")  # type: ignore
    plt.savefig(f'{date_str}-{label}.png', bbox_inches='tight')  
    plt.savefig(f'{date_str}-{label}.pdf', bbox_inches='tight')  

#%%
def sample_similarity_tsne(dataset, search_data):
    result_list = []
    x = dataset[0]
    y = dataset[1]
    for xy in zip(x, y):
        result = np.dot(list(xy), search_data) / (np.linalg.norm(list(xy)) * np.linalg.norm(search_data))
        result_list.append(result)
    return np.argmax(result_list)

def sample_similarity_bert(dataset, search_data, utterance):
    for index, s in enumerate(search_data, 1):
        result_list = []
        for i in dataset.index:
            d = dataset.loc[i]
            result = np.dot(list(d), s) / (np.linalg.norm(list(d)) * np.linalg.norm(s))
            result_list.append(result)
        print(f"クラスタリング: {index}")
        print(utterance[np.argmax(result_list)])

# %%
print("sentece_BERT")
plot_function(sentence_tsne_emnedding_df, "sentence", sentence_tnse_kmeans_model.cluster_centers_)
print("cls token")
plot_function(cls_tsne_emnedding_df, "CLS", cls_token_tnse_kmeans_model.cluster_centers_)
# %%
print("similality")
print("sentence BERT")
sample_similarity_bert(sentence_embeddings_df, sentence_kmeans_model.cluster_centers_, utterance)
# %%
print("cls token")
sample_similarity_bert(cls_token_embeddings_df, cls_token_kmeans_model.cluster_centers_, utterance)

# %%
