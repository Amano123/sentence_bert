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
utterance_embedding = model.encode(utterance, convert_to_tensor=True)
# %%
score_list = []
for INDEX, w in enumerate(utterance_embedding):
    if INDEX == 0:
        continue
    score = util.pytorch_cos_sim(utterance_embedding[INDEX], utterance_embedding[INDEX-1])
    score_list.append(score.cpu().item())

# %%
%load_ext tensorboard
import os
logs_base_dir = "runs"
os.makedirs(logs_base_dir, exist_ok=True)
# %%
import torch
from torch.utils.tensorboard import SummaryWriter
#%%
import tensorflow as tf
import tensorboard as tb
#%%
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

summary_writer = SummaryWriter()
summary_writer.add_embedding(mat=np.array(utterance_embedding.cpu()), metadata=speaker)
# %%
%tensorboard --logdir {logs_base_dir}
# %%
