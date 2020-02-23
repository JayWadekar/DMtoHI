#!/usr/bin/env python

#SBATCH --partition=p100_4,p40_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=12:30:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=20
#SBATCH -o "/scratch/dsw310/CCA/output/Reg/outReg.log"
#SBATCH -e "/scratch/dsw310/CCA/output/Reg/errReg.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: Look at LR, number of epochs, input net
#Changed: Loss criterion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import sys, time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from uNet import *
from tools import *
from data_utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dire='/scratch/dsw310/CCA/output/Reg/'

num_epochs=100
batch_size=42; num_workers=16
#eval_frequency=2

IndList2=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_original.dat'); IndList2=IndList2.astype(int)  
class SimuData2(Dataset):
    def __init__(self,lIndex,hIndex,hod=0,aug=0,test=0):
        self.datafiles = []
        self.hod=hod
        self.aug=aug
        self.test=test
        for i in np.arange(lIndex,hIndex):
            self.datafiles+=[IndList2[i]]
    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.hod,self.aug,self.test)
    def __len__(self):
        return len(self.datafiles)

best_val = 1e10; thres=0.6
mask_model = nn.DataParallel(DMUnet(BasicBlock).to(device))
mask_model.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0208_sd.pt'))
mask_model.eval()
for param in mask_model.parameters(): param.requires_grad = False

reg_model = nn.DataParallel(RegUnet(BasicBlock).to(device))
#reg_model.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Reg/0218.pt'))

#criterion = reg_loss(weight_ratio=5e-3,thres=0.87)
optimizer = torch.optim.Adam(reg_model.parameters(),lr=1e-8, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-5)
start_time = time.time()

TrainSet=SimuData2(0,165000,hod=0,aug=1)
ValSet=SimuData2(165000,180000,hod=0,aug=1)
TrainLoader=DataLoader(TrainSet, batch_size=batch_size,shuffle=True, num_workers=num_workers)
ValLoader=DataLoader(ValSet, batch_size=batch_size,shuffle=True, num_workers=num_workers)
loss_train = []; loss_val = []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    reg_model.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        inp=data[0].cuda()
        mask = (mask_model(inp) > thres)
        if mask.any()==0: continue
        Y_pred = (reg_model(inp))[mask]
        target=(data[1].cuda())[mask]
        loss =((Y_pred - target).pow(2)*torch.exp((target-thres)*8.)).mean() if mask.any()>0 else 0.
        #loss = criterion(Y_pred[mask], target[mask])
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
    np.savetxt(dire+'trainReg.dat',loss_train)
    
    reg_model.eval()
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            inp=data[0].cuda()
            mask = (mask_model(inp) > thres)
            if mask.any()==0: continue
            Y_pred = (reg_model(inp))[mask]
            target=(data[1].cuda())[mask]
            _loss =(((Y_pred - target).pow(2)*torch.exp((target-thres)*8.)).mean()).item() if mask.any()>0 else 0.
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valReg.dat',loss_val)
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        print('Saving model loss:',best_val)
        torch.save(reg_model.state_dict(),dire+'BestReg.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

