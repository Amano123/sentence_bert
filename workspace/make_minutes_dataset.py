#%%
import os
import re
from typing import List
from PoliInfo_dataclasses import load_minutes
import shutil
from PoliInfo_dataclasses.minutes import (DProceedingObject,LProceedingObject,
                                        MinutesObject)

#%%
DATASET_NAME = "PoliInfo3-FormalRun-dataset"

path_minutes_gold = f"{DATASET_NAME}/PoliInfo3_BAM-minutes-gold.json" 
path_minutes_test = f"{DATASET_NAME}/PoliInfo3_BAM-minutes-test.json"
path_minutes_training = f"{DATASET_NAME}/PoliInfo3_BAM-minutes-training.json"

# %%
# データのロード
train_all_minutes: MinutesObject = load_minutes(path_minutes_training)
# 市議会のデータ
train_local_minutes: List[LProceedingObject] = train_all_minutes.local
# 国会のデータ
train_diet_minutes: List[DProceedingObject] = train_all_minutes.diet

# データのロード
test_all_minutes: MinutesObject = load_minutes(path_minutes_test)
# 市議会のデータ
test_local_minutes: List[LProceedingObject] = test_all_minutes.local
# 国会のデータ
test_diet_minutes: List[DProceedingObject] = test_all_minutes.diet

# %%
def make_local_utterance_dataset(local_minutes: List[LProceedingObject], train_or_test: str):
    os.makedirs(f"poliinfo_utterance_dataset_test/{train_or_test}", exist_ok = True)
    shutil.rmtree(f"poliinfo_utterance_dataset_test/{train_or_test}")
    os.mkdir(f"poliinfo_utterance_dataset_test/{train_or_test}")
    for local_m in local_minutes:
        date = local_m.date
        proceeding = local_m.proceeding
        for proc in proceeding:
            speakerPosition = proc.speakerPosition
            speaker = proc.speaker
            utterance = proc.utterance
            moneyExpressions = proc.moneyExpressions
            relatedID = []
            if moneyExpressions:
                for mEx in moneyExpressions:
                    if mEx.relatedID == None:
                        continue
                    relatedID.extend(mEx.relatedID)

            utterance = utterance.replace("\n", "")
            utterance = re.split("[，、。]", utterance)
            with open(f"poliinfo_utterance_dataset_test/{train_or_test}/{date}.tsv", mode="a") as f:
                for u in utterance:
                    if u == "":
                        continue
                    f.write(f"{speakerPosition}\t{speaker}\t{u}\t{list(set(relatedID))}\n")

# %%
make_local_utterance_dataset(train_local_minutes, "train")
make_local_utterance_dataset(test_local_minutes, "test")
# %%
