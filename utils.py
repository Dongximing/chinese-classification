import torch
import transformers
from transformers import BertTokenizer
class bert_chineseDataset(torch.utils.data.Dataset):
    def __init__(self,text,label,tokenizer,max_len):
        self.text = text
        self.labels = label
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            text= self.text[item],
            add_special_tokens=True,
            max_length= self.max_len,
            return_attention_mask= True,
            return_token_type_ids= True,
            padding=False

        )
        return {'input_ids':encoding['input_ids'],'attention_mask': encoding['attention_mask'], 'token_type_ids':encoding['token_type_ids'],
                'label': self.labels[item]}
def bert_chinese_generation(training_example,training_label,validation_example,validation_label,testing_example,testing_label,tokenizer,max_length):
    return (bert_chineseDataset(training_example,training_label,tokenizer,max_length),
            bert_chineseDataset(validation_example, validation_label, tokenizer, max_length),
            bert_chineseDataset(testing_example, testing_label, tokenizer, max_length)
            )