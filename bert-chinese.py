import argparse

import torch.nn as nn
import numpy as np
import pandas as pd
import os
def data_process(train_data_path, validation_data_path,test_data_path):
    training_example = []
    training_label = []
    validation_example = []
    validation_label =[]
    testing_example = []
    testing_label = []

    with open(train_data_path) as f:
        train_lines = f.readlines()
    print(train_lines[0])

    for line in train_lines:
        example, label = line.split("\t")
        training_example.append(example)
        training_label.append(int(label))
    with open(validation_data_path) as f:
        validation_lines = f.readlines()


    for line in validation_lines:
        example, label = line.split("\t")
        validation_example.append(example)
        validation_label.append(int(label))
    with open(test_data_path) as f:
        test_lines = f.readlines()


    for line in test_lines:
        example, label = line.split("\t")
        testing_example.append(example)
        testing_label.append(int(label))

    return testing_label,test_lines,testing_example





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt')
    parser.add_argument('--valid_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/dev.txt')
    parser.add_argument('--text_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/test.txt')
    args = parser.parse_args()
    print("111")


    train_dataset,validation_dataset,test_dataset = data_process(args.train_path,args.valid_path,args.text_path)
    print(train_dataset)
    print(test_dataset)
if __name__ == "__main__":
    main()