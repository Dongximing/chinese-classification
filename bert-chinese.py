import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import transformers
from tqdm import tqdm
from transformers import BertTokenizer,BertModel,AdamW,get_linear_schedule_with_warmup
import torch.distributed as dist

import config
from utils import bert_chinese_generation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Bert_base


def data_process(train_data_path, validation_data_path,test_data_path,tokenizer,max_length):
    training_example = []
    training_label = []
    validation_example = []
    validation_label = []
    testing_example = []
    testing_label = []


    with open(train_data_path) as f:

        train_lines = f.readlines()

    for line in train_lines:


        example, label = line.split("\t")
        training_example.append(example)
        training_label.append(int(label))

    with open(validation_data_path) as f1:
        validation_lines = f1.readlines()

    for line in validation_lines:
        example, label = line.split("\t")
        validation_example.append(example)
        validation_label.append(int(label))

    with open(test_data_path) as f2:
        test_lines = f2.readlines()


    for line in test_lines:

        example, label = line.split("\t")
        testing_example.append(example)
        testing_label.append(int(label))


    return bert_chinese_generation(training_example,training_label,validation_example,validation_label,testing_example,testing_label,tokenizer,max_length)


def generate_batch(batch):

    input_ids = [torch.LongTensor(entry['input_ids']) for entry in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)
    attention_mask = [torch.LongTensor(entry['attention_mask'] )for entry in batch]
    attention_mask = pad_sequence(attention_mask,batch_first=True)
    token_type_ids = [torch.LongTensor(entry['token_type_ids'] )for entry in batch]
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
def training(local_rank,criterion,train,optimizer,model,scheduler,device):
    model.train()
    training_loss = 0
    training_acc = 0
    for i , data in tqdm(enumerate(train),total=len(train)):
        input_ids, attention_mask, token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.cuda(local_rank), attention_mask.cuda(local_rank),\
                                                           token_type_ids.cuda(local_rank), torch.LongTensor(label)
        label = label.cuda(local_rank)
        optimizer.zero_grad()
        output = model(ids=input_ids,mask=attention_mask,token_type_ids = token_type_ids)
        loss = criterion(output,label)
        acc,_ = categorical_accuracy(output,label)
        training_acc+= acc.item()
        training_loss+=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return training_loss/len(train), training_acc/len(train)
def testing(criterion,validation,model,device):
    model.eval()
    testing_loss = 0
    testing_acc = 0
    for i , data in tqdm(enumerate(validation),total=len(validation)):
        input_ids, attention_mask, token_type_ids, label = data
        input_ids, attention_mask, token_type_ids, label = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), torch.LongTensor(label)
        label = label.to(device)


        with torch.no_grad():
            output = model(ids=input_ids,mask=attention_mask,token_type_ids = token_type_ids)
        loss = criterion(output,label)
        acc,_= categorical_accuracy(output,label)


        testing_loss+=loss.item()
        testing_acc+=acc.item()

    return testing_loss/len(validation), testing_acc/len(validation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt')
    parser.add_argument('--valid_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/dev.txt')
    parser.add_argument('--test_path',type=str,default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/test.txt')
    parser.add_argument('--local_rank',type=int)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    max_length = 32
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_chinese = BertModel.from_pretrained('bert-base-chinese')
    criterion = nn.CrossEntropyLoss()
    bert_chinese_model = Bert_base(bert=bert_chinese,hidden_state=768,num_class=10)

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
        optimizer, num_warmup_steps=0, num_training_steps=int(180000/128*10)
    )
    config.seed_torch()
    n_gpus = 4
    dist.init_process_group('nccl',rank=args.local_rank,world_size=n_gpus)
    torch.cuda.set_device(args.local_rank)
    bert_chinese_model = torch.nn.parallel.DistributedDataParallel(bert_chinese_model.cuda(args.local_rank),device_ids=[args.local_rank])











    train_dataset,validation_dataset,test_dataset = data_process(args.train_path,args.valid_path,args.test_path,tokenizer,max_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train = DataLoader(train_dataset,collate_fn=generate_batch, batch_size=128,sampler=train_sampler)

    validation = DataLoader(validation_dataset,collate_fn=generate_batch,batch_size=128,shuffle=False)
    test = DataLoader(test_dataset,collate_fn=generate_batch,batch_size=128,shuffle=False)
    best_loss = float('inf')
    for epoch in range (epochs):
        train_sampler.set_epoch(epoch=epoch)
        train_loss,train_acc = training(args.local_rank,criterion,train,optimizer,bert_chinese_model,scheduler,device)
        valid_loss,valid_acc =testing(criterion,validation,bert_chinese_model,device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        if valid_loss< best_loss and args.local_rank == 0:
            best_loss = valid_loss
            torch.save(bert_chinese_model.module.state_dict(),config.bert_chinese_base_path)



    print("testing")
    bert_chinese_model.load_state_dict(torch.load(config.bert_chinese_base_path))
    test_loss, test_acc = testing(criterion, test, bert_chinese_model, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")


if __name__ == "__main__":
    main()