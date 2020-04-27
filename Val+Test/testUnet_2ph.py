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

#Notes: 140790, 138509, 188924, 32146, 1465 is interesting
# srun -t00:01:00 --mem=3GB --gres=gpu:1 --pty /bin/bash
# srun -c2 -t0:03:00 --mem=3GB --pty /bin/bash

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/home/dsw310/CCA/functions')
from uNet import *
from data_utils import *
from tools import *

thres=1.85
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mask_model = nn.DataParallel(DMUnet(BasicBlock).to(device))
mask_model.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0325.pt',map_location=device))
mask_model.eval()
for param in mask_model.parameters(): param.requires_grad = False

reg_model = nn.DataParallel(RegUnet(BasicBlock).to(device))
reg_model.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Reg/0422.pt',map_location=device))
reg_model.eval()
for param in reg_model.parameters(): param.requires_grad = False
#criterion = nn.MSELoss()

start_time = time.time()
ind=1465 

class SimuData2(Dataset):
    def __init__(self,index,aug=0,test=0):
        self.datafiles = []
        self.aug=aug
        self.test=test
        self.datafiles+=[index]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.aug,test=0)

    def __len__(self):
        return len(self.datafiles)

TrainSet=SimuData2(ind,aug=1,test=0)
TrainLoader=DataLoader(TrainSet, batch_size=1,shuffle=True, num_workers=4) 
loss_train = []
loss_val = []
num_epochs=1

for epoch in range(num_epochs):
    
    for t_val, data in enumerate(TrainLoader,0):
        with torch.no_grad():
            inp=data[0]
            Y_pred = mask_model(inp)
            mask = (Y_pred > thres)
            if mask.any()==1:
                print('Yes mask')
                Y_pred[mask]=(reg_model(inp))[mask]
            temp3=Y_pred.cpu()
            temp=data[1]
            temp2=data[1].cpu()
            #loss = criterion(Y_pred, data[1])
            loss = (((Y_pred - temp).pow(2))*(torch.exp(temp*5.)/120.)).mean()
            loss_train.append(loss.item())
            print ('Val {}'.format(loss.item()))
    #np.savetxt('/scratch/dsw310/CCA/Val+Test/valLoss.dat',loss_train)
    np.save('/scratch/dsw310/CCA/Val+Test/MLout.npy',temp3.numpy())
    np.save('/scratch/dsw310/CCA/Val+Test/Illustris.npy',temp2.numpy())
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))








