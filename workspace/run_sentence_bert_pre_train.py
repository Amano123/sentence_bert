#%%
from datetime import datetime
import math
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator, LabelAccuracyEvaluator
from sentence_bert import Data_Loader

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# %%
# 使用モデル
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
word_embedding_model = models.Transformer(model_name, max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:3")

model_save_path = 'output/training_nli_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# %%
BATCH_SIZE = 32
folder_name = "jsnli_1.1"
train_file_name = "train_w_filtering.tsv"
dev_file_name = "dev.tsv"
jsnli_reader = Data_Loader.JSNLI(folder_name)
train_num_labels = jsnli_reader.get_num_labels()

train_data = SentencesDataset(jsnli_reader.get_examples(train_file_name), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.SoftmaxLoss(model=model, 
                                sentence_embedding_dimension=pooling_model.get_sentence_embedding_dimension(), 
                                num_labels=train_num_labels)

#%%
dev_data = SentencesDataset(jsnli_reader.get_examples(dev_file_name), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=BATCH_SIZE)
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name="val")

#%%
num_epochs = 20

#%%
warmup_steps = math.ceil(len(train_dataloader) * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# %%
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=100,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          checkpoint_path = model_save_path + "/checkpoint",
          checkpoint_save_steps = 1000
          )

# %%
# test_model = SentenceTransformer("/home/yo/workspace/output/training_nli_cl-tohoku/bert-base-japanese-whole-word-masking-2022-06-11_05-14-36")
# test_model.to(model.device)
# train_loss.model = test_model
# test_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name="test")
# test_evaluator(test_model, output_path=model_save_path + "/test")

# %%
