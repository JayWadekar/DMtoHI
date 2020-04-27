#!/usr/bin/env python

#SBATCH --partition=p40_4,p100_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=14
#SBATCH -o "/scratch/dsw310/CCA/output/Unet/outUnet.log"
#SBATCH -e "/scratch/dsw310/CCA/output/Unet/errUnet.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: Look at Loss parameters, number of epochs, input net
#Changed: 
#Time 26m for 28 Batchsize, 
#gpu=8GB,cpus-per-task=17,batch_size=42; num_workers=16
#gpu=5GB,cpus-per-task=14,batch_size=28; num_workers=14

#sleep 300m && cp /scratch/dsw310/CCA/output/Unet/BestUnet.pt /scratch/dsw310/CCA/Saved/BestModel/Unet/0318.pt && sbatch main_Unet.py
# && sbatch ./Val+Test/MakeHI.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/home/dsw310/CCA/functions')
from uNet import *
from data_utils import SimuData
from tools import *


#dire='/scratch/dsw310/CCA/trash/TrashOut/'
#imp=''

dire='/scratch/dsw310/CCA/output/Unet/'
imp=''

num_epochs=200
batch_size=28; num_workers=14
#eval_frequency=2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = nn.DataParallel(DMUnet(BasicBlock).to(device))
net.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0325.pt'))
criterion = DMUnet_loss(weight_ratio=5e-3,thres=0.87)
optimizer = torch.optim.Adam(net.parameters(),lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

start_time = time.time()

TrainSet=SimuData(0,174659,aug=1)
ValSet=SimuData(174659,209481,aug=1)# Changed according to input file
TrainLoader=DataLoader(TrainSet, batch_size=batch_size,shuffle=True, num_workers=num_workers)
ValLoader=DataLoader(ValSet, batch_size=batch_size,shuffle=True, num_workers=num_workers)

loss_train = []
loss_val = []
best_val = 1e8

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    net.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        Y_pred = net(data[0].cuda())
        loss = criterion(Y_pred, data[1].cuda())
        #loss =((Y_pred - target).pow(2)*torch.exp((target-0.4)*8.)).mean()
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
    np.savetxt(dire+'trainUnet'+imp+'.dat',loss_train)
    
    net.eval()
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            Y_pred = net(data[0].cuda())
            _loss += criterion(Y_pred, data[1].cuda()).item()
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valUnet'+imp+'.dat',loss_val)
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        print('Saving model loss:',best_val)
        torch.save(net.state_dict(),dire+'BestUnet.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


