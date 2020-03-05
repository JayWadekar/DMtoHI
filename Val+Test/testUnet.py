#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutTest.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrTest.log"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: 140790, 138509, 188924, 32146 is interesting
# srun -t00:01:00 --mem=3000 --gres=gpu:1 --pty /bin/bash

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from uNet import *
from data_utils import *
from tools import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = DMUnet(BasicBlock).to(device)
criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(net.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
net = torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0208.pt')
net.eval()
start_time = time.time()
ind=140790

class SimuData2(Dataset):
    def __init__(self,index,hod=0,aug=0,test=0):
        self.datafiles = []
        self.hod=hod
        self.aug=aug
        self.test=test
        self.datafiles+=[index]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.hod,self.aug,test=0)

    def __len__(self):
        return len(self.datafiles)

TrainSet=SimuData2(ind,hod=0,aug=1,test=0)
TrainLoader=DataLoader(TrainSet, batch_size=1,shuffle=True, num_workers=4) 
loss_train = []
loss_val = []
num_epochs=1

for epoch in range(num_epochs):
    
    for t_val, data in enumerate(TrainLoader,0):
        with torch.no_grad():
            Y_pred = net(data[0].cuda())
            temp3=Y_pred.cpu()
            temp=data[1].cuda()
            temp2=data[1].cpu()
            #loss = criterion(Y_pred, data[1].cuda())
            loss = (((Y_pred - temp).pow(2))*(torch.exp(temp*10.)/120.)).mean()
            loss_train.append(loss.item())
            print ('Val {}'.format(loss.item()))
    #np.savetxt('/scratch/dsw310/CCA/Val+Test/valLoss.dat',loss_train)
    np.save('/scratch/dsw310/CCA/Val+Test/MLout.npy',temp3.numpy())
    np.save('/scratch/dsw310/CCA/Val+Test/Illustris.npy',temp2.numpy())
    original=np.load('/scratch/dsw310/CCA/data/DM+halos2/'+str(ind)+'.npy')
    np.save('/scratch/dsw310/CCA/Val+Test/HaloModel.npy',original)
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
#--------------------------------------------------------------------------------------------------------
#Below for DM + HOD
'''
for epoch in range(num_epochs):
    
    for t_val, data in enumerate(TrainLoader,0):
        with torch.no_grad():
            Y_pred = net(data[0].cuda(),data[1].cuda())
            temp3=Y_pred.cpu()
            temp=data[2].cuda()
            temp2=data[2].cpu()
            #loss = criterion(Y_pred, data[1].cuda())
            loss = (((Y_pred - temp).pow(2))*(torch.exp(temp*13.))).mean()
            loss_train.append(loss.item())
            print ('Val {}'.format(loss.item()))
    #np.savetxt('/scratch/dsw310/CCA/Val+Test/valLoss.dat',loss_train)
    np.save('/scratch/dsw310/CCA/Val+Test/MLout.npy',temp3.numpy())
    np.save('/scratch/dsw310/CCA/Val+Test/Illustris.npy',temp2.numpy())
    original=np.load('/scratch/dsw310/CCA/data/DM+halos/'+str(ind)+'.npy')
    #np.save('/scratch/dsw310/CCA/Val+Test/HaloModel.npy',original)
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
'''







