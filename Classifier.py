#!/usr/bin/env python

#SBATCH --partition=p100_4,p40_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=10:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=22
#SBATCH -o "/scratch/dsw310/CCA/output/outCla.log"
#SBATCH -e "/scratch/dsw310/CCA/output/errCla.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: Look at LR, number of epochs, input net
#Changed: 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from tools import *
from Models import *
from data_utils import SimuData


dire='/scratch/dsw310/CCA/output/'
imp=''

num_epochs=35

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
conv1_out=6; conv3_out=8; conv5_out=10; thres=0.1
weightLoss=500.; weightLoss=(torch.from_numpy(np.array([1.,weightLoss])/(1+weightLoss))).float().to(device)
model = Inception(1, conv1_out, conv3_out, conv5_out).to(device)
criterion = CrossEntropy_nn_loss(weightLoss,thres)
        
model = nn.DataParallel(model)
#model = torch.load('/scratch/dsw310/CCA/Saved/BestModel/Cla/1101.pt')

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)

start_time = time.time()

TrainSet=SimuData(0,165000,0)#0,180000
ValSet=SimuData(165000,180000,0)#180000,200000
TrainLoader=DataLoader(TrainSet, batch_size=40,shuffle=True, num_workers=20)
ValLoader=DataLoader(ValSet, batch_size=40,shuffle=True, num_workers=20)

loss_train = []
loss_val = []
best_val = 1e8

def train(model, criterion, optimizer):
    model.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        Y_pred = model(data[0].to(device))
        target=data[1].cuda()
        loss = criterion(Y_pred, target)
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
    np.savetxt(dire+'trainCla'+imp+'.dat',loss_train)

def validate(model, criterion):
    model.eval()
    global best_val
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            Y_pred = model(data[0].to(device))
            target=data[1].to(device)
            _loss += (criterion(Y_pred, target)).item()
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valCla'+imp+'.dat',loss_val)
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        print('Saved model loss is {}'.format(best_val))
        torch.save(model,dire+'BestCla'+imp+'.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
def main():
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train(model, criterion, optimizer)
        validate(model, criterion)
   
   
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()
