import os
import argparse
import config

import torch
import tqdm
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import DataProcessor
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptForClassification,PromptDataLoader
from transformers import AdamW,get_linear_schedule_with_warmup
# dataset_path = '/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/prompt/train_labels.txt'
#
# example ='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/prompt/train.txt'
# processor = DataProcessor(labels_path=dataset_path)
# #trainvalid_dataset = processor.get_train_examples(data_dir="/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/prompt/train.txt")
# print(processor.get_num_labels())
#
# trainvalid_dataset = processor.get_examples(example,'train')
# print(trainvalid_dataset[0])

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc,top_pred
def data_process(train_data_path, validation_data_path,test_data_path,res):

    training_examples = []
    validation_examples = []
    testing_examples = []


    with open(train_data_path) as f:

        train_lines = f.readlines()

    for index, line in enumerate(train_lines):
        text, label = line.split("\t")
        example = InputExample(guid=str(index),text_a=text,label= int(label))
        training_examples.append(example)

    with open(validation_data_path) as f1:
        validation_lines = f1.readlines()

    for index, line in enumerate (validation_lines):
        text, label = line.split("\t")
        example = InputExample(guid=str(index), text_a=text, label=int(label))
        validation_examples.append(example)

    with open(test_data_path) as f2:
        test_lines = f2.readlines()
    for  index, line in enumerate (test_lines):
        text, label = line.split("\t")
        example = InputExample(guid=str(index), text_a=text, label=int(label))
        testing_examples.append(example)
    return training_examples,validation_examples,testing_examples
def training(criterion,train,optimizer,model,scheduler,device):
    model.train()
    training_loss = 0
    training_acc = 0
    for i , inputs in tqdm(enumerate(train),total=len(train)):

        inputs = inputs.to(device)

        output = model(inputs)
        labels = inputs['label']
        loss = criterion(output,labels)
        acc,_ = categorical_accuracy(output,labels)

        training_acc+= acc.item()
        training_loss+=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    return training_loss/len(train), training_acc/len(train)
def testing(validation,device,criterion,model):
    model.eval()
    testing_loss = 0
    testing_acc = 0
    for i, inputs in tqdm(enumerate(validation), total=len(validation)):
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
        labels = inputs['label']
        loss = criterion(output, labels)
        acc, _ = categorical_accuracy(output, labels)
        testing_loss += loss.item()
        testing_acc += acc.item()

    return testing_loss / len(validation), testing_acc / len(validation)


def main():
    config.seed_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt')
    parser.add_argument('--valid_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/dev.txt')
    parser.add_argument('--test_path', type=str,
                        default='/home/dongxx/projects/def-parimala/dongxx/chinese/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/test.txt')
    args = parser.parse_args()
    res = {'财经':0,'房产':1,'股票':2,'教育':3,'科技':4,'社会':5,'时政':6,'体育':7,'游戏':8,'娱乐':9}
    classes = ['财经','房产','股票','教育','科技','社会','时政','体育','游戏','娱乐']
    label_words = {'财经':'金融','房产':'房地产','股票':'证劵','教育':'培训','科技':'科学技术','社会':'当今社会','时政':'政治','体育':'运动','游戏':'电子竞技','娱乐':'八卦'}
    train_dataset, validation_dataset, test_dataset = data_process(train_data_path=args.train_path,validation_data_path=args.valid_path,test_data_path=args.test_path,res=res)
    plm, tokenizer, model_config, Wrapperclass = load_plm('bert','bert-base-chinese')
    promptTemplate = ManualTemplate(text='{"placeholder":"text_a"} 它是 {"mask"} 新闻',tokenizer=tokenizer)
    promptVerbalizar = ManualVerbalizer(classes=classes,label_words=label_words,tokenizer=tokenizer)
    prompt_model = PromptForClassification(template=promptTemplate,plm=plm,verbalizer=promptVerbalizar)
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    epochs = 5
    batch_size = 128
    training_dataset = PromptDataLoader(dataset=train_dataset,max_seq_length=64,batch_size=batch_size,shuffle=True,tokenizer_wrapper_class=Wrapperclass,tokenizer=tokenizer,template=promptTemplate)
    validing_dataset = PromptDataLoader(dataset=validation_dataset,max_seq_length=64,batch_size=batch_size,shuffle=False,tokenizer_wrapper_class=Wrapperclass,tokenizer=tokenizer,template=promptTemplate)
    testing_dataset = PromptDataLoader(dataset=test_dataset,max_seq_length=64,batch_size=batch_size,shuffle=False,tokenizer_wrapper_class=Wrapperclass,tokenizer=tokenizer,template=promptTemplate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function.to(device)
    prompt_model.to(device)
    best_loss = float('inf')
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0,num_training_steps=len(training_dataset)/batch_size*epochs)
    for epoch in range (epochs):
        train_loss, train_acc= training(train = training_dataset,criterion=loss_function,optimizer=optimizer,model=prompt_model,scheduler=scheduler,device=device)
        valid_loss,valid_acc = testing (validation = validing_dataset,criterion=loss_function,model=prompt_model,device= device)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(prompt_model.state_dict(), config.bert_chinese_base_path)
    print("testing")
    prompt_model.load_state_dict(torch.load(config.bert_chinese_base_path))
    test_loss, test_acc = testing(validation=testing_dataset, criterion=loss_function , model= prompt_model,device =device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print("testing done")


if __name__ == "__main__":
    main()