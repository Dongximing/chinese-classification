import torch.nn as nn
import transformers
import torch.nn.functional as F
class Bert_base(nn.Module):
    def __init__(self,bert,hidden_state,num_class):
        super(Bert_base,self).__init__()
        self.bert = bert
        self.fc = nn.Linear(hidden_state,num_class)
    def forward(self,ids,mask, token_type_ids):
        pooled = self.bert(ids,mask,token_type_ids)[1]
        out = self.fc(pooled)
        return out

