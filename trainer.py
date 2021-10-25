import os
import argparse
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.optim as optim

from dataloader import ForestDataset
from utils import get_train_weights, get_train_weights_binary
from utils import get_acc_seg, get_acc_seg_weighted, get_acc_nzero, get_acc_class, get_acc_binseg
from models import UNet
from eq_models import UNet_eq

def main():
    
    parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
    parser.add_argument('--savedir',      dest='savedir',      type=str,     default='run0',               help='Dataset name.')
    parser.add_argument('--epochs',       dest='epochs',       type=int,     default=200,               help='Number of epochs.')
    parser.add_argument('--batch_size',   dest='batch_size',   type=int,     default=32,                help='Batch size for dataloader.')
    parser.add_argument('--lr',           dest='lr',           type=float,   default=0.001,               help='Learning rate.')
    parser.add_argument('--device',       dest='device',       type=str,     default='cuda',           help='cuda or cpu.')
    parser.add_argument('--model',        dest='model',        type=str,     default='unet',           help='unet or unet_eq.')
    args = parser.parse_args()
    print(args)
    
    if os.path.isdir(f'Outputs/{args.savedir}'):
        raise NameError(f'Dir Outputs/{args.savedir} already exists!')
    else:
        os.mkdir(f'Outputs/{args.savedir}')
    
    train_dataset = ForestDataset(csv_file='ForestNetDataset/train.csv',
                                        root_dir='ForestNetDataset')
    val_dataset = ForestDataset(csv_file='ForestNetDataset/val.csv',
                                        root_dir='ForestNetDataset')
    test_dataset = ForestDataset(csv_file='ForestNetDataset/test.csv',
                                        root_dir='ForestNetDataset')

    ## Get weights for re-weighting optimisation due to inbalance in dataset
    train_weights = get_train_weights(train_dataset)
    train_weights_bin_seg = get_train_weights_binary(train_dataset)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             shuffle=False)

    if args.model == 'unet':
        net = UNet(3, 1)
    elif args.model == 'unet_eq':
        net = UNet_eq(3, 1)
    net = net.to(args.device)
    print(net)
    
    train_weights = torch.from_numpy(train_weights).type(torch.float).to(args.device)
    criterion_class = nn.CrossEntropyLoss(weight=train_weights)
    criterion_seg = nn.BCEWithLogitsLoss(pos_weight=train_weights_bin_seg.to(args.device))
    
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    
    log_file = open(f'Outputs/{args.savedir}/training.txt','w')
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc_seg = 0.0
        running_acc_class = 0.0
        net.train()
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, segs, labels = data
            inputs = inputs.to(args.device)
            segs = segs.to(args.device)
            labels = labels.to(args.device)-1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, out_class = net(inputs)

            loss_seg = criterion_seg(outputs, torch.unsqueeze(segs.float(), dim=1))
            loss_class = criterion_class(out_class, labels)

            loss = loss_class + loss_seg

            loss.backward()
            optimizer.step()

            m1, m0, m_avg = get_acc_binseg(outputs, torch.unsqueeze(segs.float(), dim=1))
            running_acc_seg += m_avg.item()
            running_acc_class += get_acc_class(out_class, labels).item()

            running_loss += loss.item()
            
        # print statistics
        summary_str = f'Train Epoch {epoch} Loss {running_loss/len(trainloader)} Accuracy Seg {running_acc_seg/len(trainloader)} Accuracy Class {running_acc_class/len(trainloader)}'
        print(summary_str)
        log_file.write(summary_str+'\n')


        running_loss = 0.0
        running_acc_seg = 0.0
        running_acc_class = 0.0
        net.eval()
        for i, data in enumerate(valloader):
            # get the inputs
            inputs, segs, labels = data
            inputs = inputs.to(args.device)
            segs = segs.to(args.device)
            labels = labels.to(args.device)-1

            # forward
            outputs, out_class = net(inputs)

            loss_seg = criterion_seg(outputs, torch.unsqueeze(segs.float(), dim=1))
            loss_class = criterion_class(out_class, labels)

            loss = loss_class + loss_seg

            m1, m0, m_avg = get_acc_binseg(outputs, torch.unsqueeze(segs.float(), dim=1))
            running_acc_seg += m_avg.item()
            running_acc_class += get_acc_class(out_class, labels).item()

            running_loss += loss.item()

        # print statistics
        summary_str = f'Val Epoch {epoch} Loss {running_loss/len(valloader)} Accuracy Seg {running_acc_seg/len(valloader)} Accuracy Class {running_acc_class/len(valloader)}'
        print(summary_str)
        log_file.write(summary_str+'\n')

        if (epoch%10 == 0) and epoch!=0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'Outputs/{args.savedir}/model_{epoch}.pt')

    print('Finished Training')
    log_file.close()

if __name__ == "__main__":
    main()  