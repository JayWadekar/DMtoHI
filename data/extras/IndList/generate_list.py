#!/usr/bin/env python
#Run on terminal
#srun -c2 -t3:00:00 --mem=4GB --pty /bin/bash
#Variables: Halo mass cutoff: 0.7 and probability of finding: 1-0.9

#0.8% for delta>0.7, 0.05% delta>0.85, 0.01% for delta>0.9
import numpy as np

outDir='/scratch/dsw310/CCA/data/smoothed/HI/'

IndList = []
for i in np.arange(0,3969*22):
    IndList+=[i]

  
for i in np.arange(3969*22,3969*63):
    if( ((i%3969)//63)<22):
        IndList+=[i]
    else:
        if(i%63<22):
            IndList+=[i]
IndList=np.array(IndList)

#np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_original.dat',IndList)

haloList=[]
haloList2=[]
for i in IndList:
    inp=np.amax(np.load(outDir+str(i)+'.npy'))
    if(inp>.7):    haloList+=[i]
    if(inp>.85):    haloList2+=[i]


np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/halo_del>0.7.dat',haloList)
np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/halo_del>0.85.dat',haloList2)
#len(haloList)*1./len(IndList)
IndList=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_original.dat')
haloList=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/halo_del>0.7.dat')
haloList2=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/halo_del>0.85.dat')

#-------------------------------------------------------------------------------
IndList=IndList[:(len(IndList)*9/10)]; haloList=haloList[:(len(haloList)*9/10)]; haloList2=haloList2[:(len(haloList2)*9/10)]

temp=haloList
for i in range(5):
    haloList=np.append(haloList,temp)

temp=haloList2
for i in range(40):
    haloList2=np.append(haloList2,temp)

IndList=np.append(IndList, np.append(haloList2,haloList))
mainList=IndList
#-------------------------------------------------------------------------------
IndList=IndList[(len(IndList)*9/10):]; haloList=haloList[(len(haloList)*9/10):]; haloList2=haloList2[(len(haloList2)*9/10):]


temp=haloList
for i in range(5):
    haloList=np.append(haloList,temp)

temp=haloList2
for i in range(40):
    haloList2=np.append(haloList2,temp)

IndList=np.append(IndList, np.append(haloList2,haloList))
mainList=np.append(mainList,IndList)
np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_haloMix.dat',mainList)
#-------------------------------------------------------------------------------


#---------------------------------------------------------------
#----------------------------------
#-- Generating list for 2-phase 

import numpy as np

thres=1.9
IndList2=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_original.dat'); IndList2=IndList2.astype(int)
listi=[]

for ind in IndList2:
    ind2=ind // 3969; ind1= (ind%3969) // 63; ind0=ind % 63
    HI=HIdata[ind2*8+4:ind2*8+12,ind1*8+4:ind1*8+12,ind0*8+4:ind0*8+12]
    if(np.amax(HI)
    
    
#--(Should generate from Illustris and not from the Unet itself?)

# srun -t01:30:00 --mem=3000 --gres=gpu:1 --pty /bin/bash

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import sys, time
sys.path.insert(0, '/home/dsw310/CCA/functions')
from uNet import *
from tools import *
from data_utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


mask_model = nn.DataParallel(DMUnet(BasicBlock).to(device))
mask_model.load_state_dict(torch.load('/scratch/dsw310/CCA/Saved/BestModel/Unet/0208_sd.pt'))
mask_model.eval()
for param in mask_model.parameters(): param.requires_grad = False

TestSet=SimuData2(0,180000,hod=0,aug=0,test=1) #180000
TestLoader=DataLoader(TestSet, batch_size=1,shuffle=False, num_workers=1)

start_time = time.time()
listi=[]
for t, data in enumerate(TestLoader,0):
    with torch.no_grad():
        inp=data[0].cuda()
        Y_pred = mask_model(inp)
        mask = (Y_pred > thres)
        if mask.any()==1:
            listi.append(data[1].numpy()[0])

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/2phase_0208Unet.dat',listi)
#---------------------------------------------------------------
#-----------------------Old and wrong below---------------------
#---------------------------------------------------------------
it=0
it2=0
lhalo=len(haloList); lhalo2=len(haloList2)
IndList = []
for i in np.arange(0,3969*22):
    a=np.random.rand()
    if a < .8:
        IndList+=[i]
    elif (a < .95):
        IndList+=[haloList2[it2]]; it2+=1; it2=it2%lhalo2; IndList+=[i]
    else:
        IndList+=[haloList[it]]; it+=1; it=it%lhalo; IndList+=[i]


for i in np.arange(3969*22,3969*63):
    a=np.random.rand()
    if( ((i%3969)//63)<22):
        if a < .8:
            IndList+=[i]
        elif (a < .95):
            IndList+=[haloList2[it2]]; it2+=1; it2=it2%lhalo2; IndList+=[i]
        else:
            IndList+=[haloList[it]]; it+=1; it=it%lhalo; IndList+=[i]
    else:
        if(i%63<22):
            if a < .8:
                IndList+=[i]
            elif (a < .95):
                IndList+=[haloList2[it2]]; it2+=1; it2=it2%lhalo2; IndList+=[i]
            else:
                IndList+=[haloList[it]]; it+=1; it=it%lhalo; IndList+=[i]

np.savetxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_haloMix.dat',IndList)




