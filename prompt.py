
import os
from openprompt.data_utils.text_classification_dataset import DataProcessor

base_path = '/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/'
dataset_name = "prompt"
dataset_path = os.path.join(base_path, dataset_name)
print(dataset_path)
processor = DataProcessor[dataset_name.lower()]()
trainvalid_dataset = processor.get_train_examples(dataset_path)
assert processor.get_num_labels() == 2
print(trainvalid_dataset[0])