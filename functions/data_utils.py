import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py 

inpDir='/scratch/dsw310/CCA/data/DM_training/'
#outDir='/scratch/dsw310/CCA/data/smoothed/HI2/'

HIdata=np.load('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_512Res_rescaled.npy')

IndList=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_haloMix.dat').astype(int)
   
class SimuData(Dataset):
    def __init__(self,lIndex,hIndex,aug=0,test=0):
        self.datafiles = []
        self.aug=aug
        self.test=test
        for i in np.arange(lIndex,hIndex):
            self.datafiles+=[IndList[i]]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.aug,self.test)

    def __len__(self):
        return len(self.datafiles)
        
def get_mini_batch(ind,aug,test):
    inp=np.load(inpDir+str(ind)+'.npy')
    if test:
        return torch.from_numpy(inp).float(),ind
    else:
        ind2=ind // 3969; ind1= (ind%3969) // 63; ind0=ind % 63
        out=np.expand_dims(HIdata[ind2*8+4:ind2*8+12,ind1*8+4:ind1*8+12,ind0*8+4:ind0*8+12],axis=0)
        if(aug==1):
            if np.random.rand() < .5:
                inp=inp[:,::-1,:,:]
                out=out[:,::-1,:,:]
            if np.random.rand() < .5:
                inp=inp[:,:,::-1,:]
                out=out[:,:,::-1,:]
            if np.random.rand() < .5:
                inp=inp[:,:,:,::-1]
                out=out[:,:,:,::-1]
            prand = np.random.rand()
            if prand < 1./6:
                inp = np.transpose(inp, axes = (0,2,3,1))
                out = np.transpose(out, axes = (0,2,3,1))
            elif prand < 2./6:
                inp = np.transpose(inp, axes = (0,2,1,3))
                out = np.transpose(out, axes = (0,2,1,3))
            elif prand < 3./6:
                inp = np.transpose(inp, axes = (0,1,3,2))
                out = np.transpose(out, axes = (0,1,3,2))
            elif prand < 4./6:
                inp = np.transpose(inp, axes = (0,3,1,2))
                out = np.transpose(out, axes = (0,3,1,2))
            elif prand < 5./6:
                inp = np.transpose(inp, axes = (0,3,2,1))
                out = np.transpose(out, axes = (0,3,2,1))
        return torch.from_numpy(inp.copy()).float(),torch.from_numpy(out.copy()).float()
        
