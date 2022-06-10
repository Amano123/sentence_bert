#%%
import sentence_transformers
from sentence_bert import Data_Loader

# %%
folder_name = "jsnli_1.1"
file_name = "train_w_filtering.tsv"
jsnli = Data_Loader.JSNLI(folder_name)
# %%
dataset = jsnli.get_examples(file_name)
# %%
