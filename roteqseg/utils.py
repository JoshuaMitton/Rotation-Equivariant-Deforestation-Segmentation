import torch
import torch.nn.functional as F
import numpy as np

## Reweighting optimisation
def get_train_weights(train_dataset):
    train_weights = [0, 0, 0, 0, 0]
    for i in range(len(train_dataset)):
        image, seg, labels = train_dataset[i]
        segs = torch.mul(seg, labels)
        indx, counts = torch.unique(segs, return_counts=True)
        for ind, count in zip(indx, counts):
            train_weights[ind] += count
            
    weights = 1 / np.asarray(train_weights[1:])
    weights = weights / np.max(weights)
    return weights

def get_train_weights_binary(train_dataset):
    train_weights_bin_seg = [0, 0]
    for i in range(len(train_dataset)):
        image, seg, labels = train_dataset[i]
        indx, counts = torch.unique(seg, return_counts=True)
        for ind, count in zip(indx, counts):
            train_weights_bin_seg[ind] += count
    train_weights_bin_seg = train_weights_bin_seg[0]/train_weights_bin_seg[1]
    return train_weights_bin_seg

## ----------------------------------------------------------------
## Accuracy functions
def get_acc_seg(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==segs)
    acc = acc.view(-1)
    return acc.sum()/len(acc)

def get_acc_seg_weighted(outputs, segs):
    outputs = torch.max(outputs, dim=1)[1]
    acc = []
    for i in range(5):
        acc_temp = (outputs==segs)
        acc_temp = acc_temp.view(-1)
        acc.append(acc_temp.sum()/len(acc_temp))
    return torch.mean(torch.stack(acc))

def get_acc_nzero(outputs, segs):
    mask = ~segs.eq(0)
    outputs = torch.max(outputs, dim=1)[1]
    acc = torch.masked_select((outputs==segs), mask)
    return acc.sum()/len(acc)

def get_acc_class(outputs, labels):
    outputs = torch.max(outputs, dim=1)[1]
    acc = (outputs==labels)
    return acc.sum()/len(acc)

def get_acc_binseg(outputs, segs):
    outputs = F.sigmoid(outputs)
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    acc = (outputs==segs)
    acc_1 = acc[segs==1].view(-1)
    acc_0 = acc[segs==0].view(-1)
    acc_1 = acc_1.sum()/len(acc_1)
    acc_0 = acc_0.sum()/len(acc_0)
    return acc_1, acc_0, (acc_1+acc_0)/2
    