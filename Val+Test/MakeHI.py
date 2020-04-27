#!/usr/bin/env python

#SBATCH --partition=p1080_4,v100_sxm2_4,p100_4,p40_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=3GB
#SBATCH --cpus-per-task=2
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutMake.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrMake.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#srun -t00:25:00 --mem=3000 --gres=gpu:1 --pty /bin/bash

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

#dire='/scratch/dsw310/CCA/data/output/HIboxes/'
side_half=20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = nn.DataParallel(DMUnet(BasicBlock).to(device))
net.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0425.pt'))
net.eval()
start_time = time.time()

class SimuData2(Dataset):
    def __init__(self,index,aug,test):
        self.datafiles = []
        self.test=test; self.aug=aug
        
        for x in range(len(index)):
            self.datafiles+=[index[x]]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.aug,self.test)

    def __len__(self):
        return len(self.datafiles)

start=42*3969+42*63+42 #32146# 19716 
ind=[]

for i in range(-side_half,side_half+1):
    for j in range(-side_half,side_half+1):
        for k in range(-side_half,side_half+1):
            ind.append(start+i*3969+j*63+k)

TestSet=SimuData2(ind,aug=0,test=1)
TestLoader=DataLoader(TestSet, batch_size=1,shuffle=False, num_workers=1)
temp=[]
    
for t, data in enumerate(TestLoader,0):
    with torch.no_grad():
        Y_pred = net(data[0].cuda())
        temp.append([data[1].numpy()[0],Y_pred.cpu()[0,0].numpy()])

np.save('/scratch/dsw310/CCA/data/output/HIboxes.npy',temp)
    
time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))







