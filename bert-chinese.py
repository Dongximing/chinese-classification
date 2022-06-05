import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import transformers
from tqdm import tqdm
from transformers import BertTokenizer,BertModel,AdamW,get_linear_schedule_with_warmup


import config
from utils import bert_chinese_generation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Bert_base


def data_process(train_data_path, validation_data_path,test_data_path,tokenizer,max_length):
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

    return bert_chinese_generation(training_example,training_label,validation_example,validation_label,testing_example,testing_label,tokenizer,max_length)


def generate_batch(batch):
    for entry in batch:
        print(entry['input_ids'])
    input_ids = [torch.Tensor(entry['input_ids']) for entry in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)
    attention_mask = [torch.Tensor(entry['attention_mask'] )for entry in batch]
    attention_mask = pad_sequence(attention_mask,batch_first=True)
    token_type_ids = [torch.Tensor(entry['token_type_ids'] )for entry in batch]
    token_type_ids = pad_sequence(token_type_ids,batch_first=True)
    label = [entry['label'] for entry in batch]
    return input_ids,attention_mask,token_type_ids,label

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc,top_pred
def training(criterion,train,optimizer,model,scheduler,device):
    model.train()
    training_loss = 0
    training_acc = 0
    for i , data in tqdm(enumerate(train),total=len(train)):
        input_ids, attention_mask, token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), torch.LongTensor(label)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(ids=input_ids,mask=attention_mask,token_type_ids = token_type_ids)
        loss = criterion(output,label)
        acc = categorical_accuracy(output,label)
        training_acc+= acc.item()
        training_loss+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return training_loss/len(train), training_acc/len(train)
def testing(criterion,validation,model,device):
    model.eval()
    test_loss = 0
    test_acc = 0
    for i , data in tqdm(enumerate(validation),total=len(validation)):
        input_ids, attention_mask, token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), torch.LongTensor(label)
        label = label.to(device)
        with torch.no_grad():
            output = model(ids=input_ids,mask=attention_mask,token_type_ids = token_type_ids)
        loss = criterion(output,label)
        acc = categorical_accuracy(output,label)
        test_acc+=loss.item()
        test_acc+=acc.item()

    return test_loss/len(validation), test_acc/len(validation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt')
    parser.add_argument('--valid_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/dev.txt')
    parser.add_argument('--text_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/test.txt')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    max_length = 32
    epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_chinese = BertModel.from_pretrained('bert-base-chinese')
    criterion = nn.CrossEntropyLoss()
    bert_chinese_model = Bert_base(bert=bert_chinese,hidden_state=768,num_class=10)
    bert_chinese_model.to(device)
    param_optimizer = list(bert_chinese_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=int(20000/8*5)
    )
    config.seed_torch()








    train_dataset,validation_dataset,test_dataset = data_process(args.train_path,args.valid_path,args.text_path,tokenizer,max_length)
    train = DataLoader(train_dataset,collate_fn=generate_batch, batch_size=128,shuffle=True)
    validation = DataLoader(validation_dataset,collate_fn=generate_batch,batch_size=128,shuffle=False)
    test = DataLoader(train_dataset,collate_fn=generate_batch,batch_size=128,shuffle=False)
    best_loss = float('inf')
    for epoch in range (epochs):
        train_loss,train_acc = training(criterion,train,optimizer,bert_chinese_model,scheduler,device)
        valid_loss,valid_acc =testing(criterion,validation,bert_chinese_model,device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss< best_loss:
            best_loss = valid_loss
            torch.save(bert_chinese_model.state_dict(),config.bert_chinese_base_path)



    print("testing")
    bert_chinese_model.load_state_dict(torch.load(config.bert_chinese_base_path))
    test_loss, test_acc = testing(criterion, test, bert_chinese_model, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")

    print(test_dataset)
if __name__ == "__main__":
    main()