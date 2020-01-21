#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Out.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Err.log"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: 32146 is interesting, 19716

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from uNet import *
from data_utils import SimuData
from tools import *


net = VanillaUnet(BasicBlock)
net.cuda()
criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(net.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
net = torch.load('/scratch/dsw310/CCA/Saved/BestModel/Reg/1031.pt')
net.eval()
start_time = time.time()
ind=19716
TrainSet=SimuData(ind,ind+1)#
#ValSet=SimuData(100,150)#26000,26100
TrainLoader=DataLoader(TrainSet, batch_size=1,shuffle=True, num_workers=4) 
#ValLoader=DataLoader(ValSet, batch_size=50,shuffle=True, num_workers=4)
loss_train = []
loss_val = []
num_epochs=1

for epoch in range(num_epochs):
    
    for t_val, data in enumerate(TrainLoader,0):
        with torch.no_grad():
            Y_pred = net(data[0].cuda(),data[1].cuda())
            temp3=Y_pred.cpu()
            temp=data[2].cuda()
            temp2=data[2].cpu()
            #loss = criterion(Y_pred, data[1].cuda())
            loss = (((Y_pred - temp).pow(2))*(torch.exp(temp*10.))).mean()
            loss_train.append(loss.item())
            print ('Val {}'.format(loss.item()))
    #np.savetxt('/scratch/dsw310/CCA/Val+Test/valLoss.dat',loss_train)
    np.save('/scratch/dsw310/CCA/Val+Test/MLout.npy',temp3.numpy())
    np.save('/scratch/dsw310/CCA/Val+Test/Illustris.npy',temp2.numpy())
    original=np.load('/scratch/dsw310/CCA/data/DM+halos/'+str(ind)+'.npy')
    np.save('/scratch/dsw310/CCA/Val+Test/HaloModel.npy',original)
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))





















