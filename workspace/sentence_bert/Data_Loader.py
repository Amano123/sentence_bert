
import os
import csv
from . import Input_Format

class JSNLI(object):
    """
    日本語SNLI(JSNLI)データセットをロードする関数
    """
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def get_examples(self, file_name, max_examples=0):
        sentence1 = []
        sentence2 = []
        labels = []
        with open(self.folder_name + "/" + file_name, encoding='utf-8', newline='') as f:
            for line in csv.reader(f, delimiter='\t'):
                labels.append(line[0])
                sentence1.append(line[1])
                sentence2.append(line[2])

        examples = []
        id = 0
        for s1, s2, label in zip(sentence1, sentence2, labels):
            guid = "%s-%d" % (file_name, id)
            id += 1
            examples.append(Input_Format(guid=guid, texts=[s1, s2], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]

class poliinfo_uterance(object):
    def __init__(self, folder_naem) -> None:
        self.folder_name = folder_naem

    def get_utterance(self, test_or_train: str, filename, max=0):
        utterance = []
        speaker = []
        labels = []
        with open(f"{self.folder_name}/{test_or_train}/{filename}", newline="") as f:
            for line in csv.reader(f, delimiter='\t'):
                utterance.append(line[2])
                speaker.append(line[1])
        return utterance, speaker
 