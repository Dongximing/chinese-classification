list = ['a','b']
res = {val: index+1 for index,val in enumerate(list)}
print(res)
import torch
# inputs = torch.FloatTensor([0,1,0,0,0,1])
# outputs = torch.LongTensor([0,2])
# inputs = inputs.view((2,3))
# outputs = outputs.view((2))
# print(inputs)
# print(outputs)
# weight_CE = torch.FloatTensor([1,1,1])
# weight_CE1 = torch.FloatTensor([2,1,1])
# ce = torch.nn.CrossEntropyLoss(weight=weight_CE)
# de1 =torch.nn.CrossEntropyLoss(reduction="none")
# dee2 = torch.nn.CrossEntropyLoss(weight=weight_CE1,reduction="none")
# loss = ce(inputs,outputs)
# loss1 = de1(inputs,outputs)
# loss2 = dee2(inputs,outputs)
# print(loss)
# print(loss1)
# print(loss2)
# y_true = [2,1,0]
# y_pred_1 = [[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]]
# y_pred_2 = [[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]]
# y_true = torch.LongTensor(y_true)
# y_pred_1= torch.FloatTensor(y_pred_1)
# y_pred_2 = torch.FloatTensor(y_pred_2)
# loss = de1(y_pred_1,y_true)
# print(loss)
# log = torch.nn.LogSoftmax(dim=1)
# nn = torch.nn.NLLLoss(reduction="none")
# # print(log(y_pred_1))
# # print(nn(log(y_pred_1),y_true))
# print(de1(y_pred_1,y_true))
# print(dee2(y_pred_1,y_true))
# a = [[0.9,0.1]]
#
# b = torch.FloatTensor(a)
# b = torch.log(b)
# print(b)
# c = [0]
# d = torch.LongTensor(c)
# f = nn(b,d)
# print(f)
# a = [[1,1.1,1.2],[0.8,1.7,1.5],[0.9,1.5,1.1]]
# c = torch.FloatTensor(a)
# m = torch.nn.Sigmoid()
# print(m(c))
# print(torch.transpose(c,1,0))
inputs = torch.tensor([[1.0,1.0,0.0,0.0],[0.0,0.0,1.0,1.0],[1.0,0.0,0.0,1.0]])
#inputs = torch.tensor([[1.0,1.0,0.0,0.0]])
#
# print(inputs.shape)
# targets = torch.LongTensor([0,1,0])
#
# w1 = torch.tensor([[0.5,0.7,0.6],[0.5,0.4,0.6],[0.4,0.9,1.0],[0.4,0.8,0.5]],requires_grad=True)
#
# f1 = inputs@w1
# sig = torch.nn.Sigmoid()
# f1_sig = sig(f1)
# f1_sig = f1_sig.requires_grad_(True)
# w2 = torch.tensor([[0.6,0.9,0.1],[0.1,0.0,0.8],[0.8,0.9,1.0]],requires_grad=True)
# f2 = f1_sig@w2
#
# f2_sig = sig(f2)
# f2_sig = f2_sig.requires_grad_(True)
# w3 = torch.tensor([[0.8,0.5],[0.8,0.1],[0.6,0.1]],requires_grad=True)
# f3 = (f2_sig@w3)
#
# st = torch.nn.Softmax(dim=1)
# print(f3)
#
# real_target = torch.Tensor([[1,0],[0,1],[1,0]])
# print(st(f3))
# print(((st(f3)-real_target).transpose(1,0))@f2_sig)
#
# loss = torch.nn.CrossEntropyLoss(reduction='none')
# total_loss = loss(f3,targets)
# print(total_loss)
#
# total_loss.backward(torch.ones_like(targets))
# print(w3.grad)
# sig_g = f2_sig*(1-f2_sig)
# sig_g1=((st(f3)-real_target))@(w3.transpose(1,0))
# print(sig_g)
# print(sig_g1)
# sig_g12=sig_g1*sig_g
# print(sig_g12)
# print((sig_g12.transpose(1,0)@f1_sig))
# print(w2.grad)
import torch
a = torch.randn(2,3)
print(a)
print(torch.mean(a))
print(torch.std(a,unbiased=False))