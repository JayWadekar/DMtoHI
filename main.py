#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=05:20:00
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=12
#SBATCH -o "/scratch/dsw310/CCA/output/output1.log"
#SBATCH -e "/scratch/dsw310/CCA/output/error1.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: lr decay changed.

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

num_epochs=18
#eval_frequency=2

net = Lpt2NbodyNet(BasicBlock)
net = nn.DataParallel(net)
net.cuda()
net = torch.load('/scratch/dsw310/CCA/output/BestModel_save.pt')#
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)

start_time = time.time()

TrainSet=SimuData(3000,30000)#23500
ValSet=SimuData(0,3000)#26000,26100
TrainLoader=DataLoader(TrainSet, batch_size=20,shuffle=True, num_workers=4)
ValLoader=DataLoader(ValSet, batch_size=20,shuffle=True, num_workers=4)

loss_train = []
loss_val = []
iterTime = 0
best_val = 10.

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    net.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
        Y_pred = net(NetInput)
        loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        print ('Test {}'.format(loss.item()))
    np.savetxt(dire+'trainLoss'+imp+'.dat',loss_train)
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    net.eval()
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
            Y_pred = net(NetInput)
            _loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).item()
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valLoss'+imp+'.dat',loss_val)
    print ('Val {}'.format(loss_val))
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        torch.save(net,dire+'BestModel'+imp+'.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    iterTime+=1
    
time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))











