#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=9:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=24
#SBATCH -o "/scratch/dsw310/CCA/output/output1.log"
#SBATCH -e "/scratch/dsw310/CCA/output/error1.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: Look at LR, number of epochs, input net
#Changed: 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/uNet')
from uNet import BasicBlock, Lpt2NbodyNet, crop_tensor
from data_utils import SimuData


#dire='/scratch/dsw310/CCA/trash/TrashOut/'
#imp=''

dire='/scratch/dsw310/CCA/output/'
imp='1'

num_epochs=11
#eval_frequency=2

net = Lpt2NbodyNet(BasicBlock)
net.cuda()
net = nn.DataParallel(net)
net = torch.load('/scratch/dsw310/CCA/Saved/BestModel/1012_2.pt')
#criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=5e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)

start_time = time.time()

TrainSet=SimuData(0,165000)#0,180000
ValSet=SimuData(165000,180000)#180000,200000
TrainLoader=DataLoader(TrainSet, batch_size=30,shuffle=True, num_workers=8)
ValLoader=DataLoader(ValSet, batch_size=30,shuffle=True, num_workers=8)

loss_train = []
loss_val = []
best_val = 1e8

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    net.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        NetInput = data[0].cuda()
        Y_pred = net(NetInput)
        temp=data[1].cuda()
        #loss = criterion(Y_pred, data[1].cuda())
        loss = (Y_pred-temp).pow(6).mean()
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        print (loss.item())
    np.savetxt(dire+'trainLoss'+imp+'.dat',loss_train)
    
    net.eval()
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            NetInput = data[0].cuda()
            Y_pred = net(NetInput)
            temp=data[1].cuda()
            #_loss += criterion(Y_pred, data[1].cuda()).item()
            _loss += ((Y_pred-temp).pow(6).mean()).item()
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valLoss'+imp+'.dat',loss_val)
    print ('Val {}'.format(loss_val))
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        torch.save(net,dire+'BestModel'+imp+'.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


