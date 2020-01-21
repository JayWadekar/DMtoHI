#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=22
#SBATCH -o "/scratch/dsw310/CCA/output/output1.log"
#SBATCH -e "/scratch/dsw310/CCA/output/error1.log"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsw310@nyu.edu

#Notes: Look at LR, number of epochs, input net, model_idx
#Changed: 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from uNet import BasicBlock, VanillaUnet, crop_tensor
from data_utils import SimuData


#dire='/scratch/dsw310/CCA/trash/TrashOut/'
#imp=''

dire='/scratch/dsw310/CCA/output/'
imp='1'

num_epochs=4

if target_class == 0:
    model = VanillaUnet(BasicBlock)
    #model = torch.load('/scratch/dsw310/CCA/Saved/BestModel/1018.pt')
    criterion = nn.CrossEntropyLoss(weight=(torch.from_numpy(np.array([1,weight_loss])/(1+weight_loss))).float())
else:
    model = two_phase_conv(mask_model,pred_model)
    #model = torch.load('/scratch/dsw310/CCA/Saved/BestModel/1018.pt')
    criterion = weighted_nn_loss(loss_weight)
        
model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(net.parameters(),lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)

start_time = time.time()

TrainSet=SimuData(0,150000)#0,180000
ValSet=SimuData(150000,165000)#180000,200000
TrainLoader=DataLoader(TrainSet, batch_size=30,shuffle=True, num_workers=7)
ValLoader=DataLoader(ValSet, batch_size=30,shuffle=True, num_workers=7)

weight_loss=20.
thres=0.1

loss_train = []
loss_val = []
best_val = 1e8

def train(model, criterion, optimizer, epoch, target_class):
    model.train()
    for t, data in enumerate(TrainLoader, 0):
        optimizer.zero_grad()
        Y_pred = model(data[0].to(device),data[1].to(device))
        if target_class == 0:
            target =(data[2] > thres).to(device).long()
        elif target_class == 1:
            target = data[2].to(device).float()
        loss = criterion(Y_pred, target)
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
    np.savetxt(dire+'trainLoss'+imp+'.dat',loss_train)

def validate(model, criterion, epoch, target_class, save_name):
    model.eval()
    _loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            Y_pred = model(data[0].cuda(),data[1].cuda())
            target=data[2].cuda()
            _loss += ((((Y_pred - target).pow(2))*(torch.exp(target*10.))).mean()).item()
    loss_val.append(_loss/(t_val+1))
    np.savetxt(dire+'valLoss'+imp+'.dat',loss_val)
    if( _loss/(t_val+1) < best_val):
        best_val= _loss/(t_val+1)
        torch.save(model,dire+'BestModel'+imp+'.pt')
    time_elapsed = time.time() - start_time
    print('Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train(model, criterion, optimizer)
        validate(model, criterion)
    
    
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()
