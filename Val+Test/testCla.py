#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import time
sys.path.insert(0, '/scratch/dsw310/CCA/functions')
from tools import *
from Models import *
from data_utils import SimuData


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
conv1_out=6; conv3_out=8; conv5_out=10; thres=0.1
weightLoss=1000.; weightLoss=(torch.from_numpy(np.array([1,weightLoss])/(1+weightLoss))).float().to(device)
model = Inception(1, conv1_out, conv3_out, conv5_out).to(device)
model = torch.load('/scratch/dsw310/CCA/Saved/BestModel/Cla/1102.pt')
model.eval()
criterion = CrossEntropy_nn_loss(weightLoss,thres)

ind=32146
ValSet=SimuData(ind,ind+1,0)
ValLoader=DataLoader(ValSet, batch_size=1,shuffle=True, num_workers=1)

loss_train = []
loss_val = []

def validate(model, criterion):
    loss=0
    for t_val, data in enumerate(ValLoader,0):
        with torch.no_grad():
            Y_pred = model(data[0].to(device))
            target=data[1].to(device)
            loss = (criterion(Y_pred, target)).item()
            Y_pred = F.softmax(Y_pred, dim=1)[:,1,:,:,:]
            Y_pred = (Y_pred > 0.5).float()
            print ('Val {}'.format(loss))
    return Y_pred



temp3=validate(model, criterion)
np.save('/scratch/dsw310/CCA/Val+Test/ClaOut.npy',temp3.cpu().numpy())


original=np.load('/scratch/dsw310/CCA/data/HI/'+str(ind)+'.npy')
np.save('/scratch/dsw310/CCA/Val+Test/Illustris.npy',np.expand_dims(original,axis=0))
#original=np.load('/scratch/dsw310/CCA/data/DM+halos/'+str(ind)+'.npy')
#np.save('/scratch/dsw310/CCA/Val+Test/HaloModel.npy',original)





