
import os
from openprompt.data_utils.text_classification_dataset import DataProcessor

dataset_path = '/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/prompt/train_labels.txt'


processor = DataProcessor(labels_path=dataset_path)
trainvalid_dataset = processor.get_train_examples(data_dir="/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/prompt/train.txt")
assert processor.get_num_labels() == 2
print(trainvalid_dataset[0])