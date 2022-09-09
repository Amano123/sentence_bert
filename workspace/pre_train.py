#%%
from datetime import datetime
import math
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator, 
    LabelAccuracyEvaluator
    )

import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# 自作関数                    
from sentence_bert import Data_Loader




# %%

# 使用モデル
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

word_embedding_model = models.Transformer(
    model_name, 
    max_seq_length=256
    )

#------------------------------------------------

# model_type = "CLS" 
# device = "cuda:3"
# pooling_model = models.Pooling(
#     word_embedding_dimension = word_embedding_model.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens = False, # 各単語ベクトルの平均
#     pooling_mode_cls_token = True, # [CLS] token
#     pooling_mode_max_tokens = False # 各単語ベクトルの最大値
#     )

# model_type = "mean" 
# device = "cuda:1"
# pooling_model = models.Pooling(
#     word_embedding_dimension = word_embedding_model.get_word_embedding_dimension(),
#     pooling_mode_mean_tokens = True, # 各単語ベクトルの平均
#     # pooling_mode_cls_token = False, # [CLS] token
#     # pooling_mode_max_tokens = False # 各単語ベクトルの最大値
#     )


model_type = "max" 
device = "cuda:2"
pooling_model = models.Pooling(
    word_embedding_dimension = word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens = False, # 各単語ベクトルの平均
    pooling_mode_cls_token = False, # [CLS] token
    pooling_mode_max_tokens = True # 各単語ベクトルの最大値
    )

#-----------------------------------------------

model: SentenceTransformer = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], 
    device=device # GPU指定
    )



model_save_path = f"output/{ model_type }_training_nli_{ model_name }_{ datetime.now().strftime('%Y-%m-%d_%H-%M-%S') }"

# %%
BATCH_SIZE: int = 32
dataset_folder_name: str = "jsnli_1.1"
dataset_train_file_name: str = "train_w_filtering.tsv"
dataset_dev_file_name: str = "dev.tsv"

jsnli_reader: object = Data_Loader.Jsnli_Data_Format(dataset_folder_name)
train_num_labels: int = jsnli_reader.get_num_labels()

train_data = SentencesDataset(
    examples = jsnli_reader.get_examples(dataset_train_file_name), 
    model = model
    )

train_dataloader = DataLoader(
    dataset = train_data, 
    shuffle = True, 
    batch_size = BATCH_SIZE
    )

train_loss = losses.SoftmaxLoss(
    model = model, 
    sentence_embedding_dimension = pooling_model.get_sentence_embedding_dimension(), 
    num_labels = train_num_labels
    )

#%%
dev_data = SentencesDataset(
    examples = jsnli_reader.get_examples(dataset_dev_file_name), 
    model = model
    )

dev_dataloader = DataLoader(
    dataset = dev_data, 
    shuffle = False, 
    batch_size = BATCH_SIZE
    )

evaluator = LabelAccuracyEvaluator(
    dataloader = dev_dataloader, 
    softmax_model = train_loss, 
    name = "val"
    )

#%%
num_epochs = 20

#%%
warmup_steps = math.ceil(len(train_dataloader) * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# %%
model.fit(
    train_objectives = [ (train_dataloader, train_loss) ],
          evaluator = evaluator,
          epochs = num_epochs,
          evaluation_steps = 100,
          warmup_steps = warmup_steps,
          output_path = model_save_path,
          checkpoint_path = model_save_path + "/checkpoint",
          checkpoint_save_steps = 1000
          )

def main():
    pass

if __name__ == "__main__":
    pass