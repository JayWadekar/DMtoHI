import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py 

inpDir='/scratch/dsw310/CCA/data/DM+halos2/'
outDir='/scratch/dsw310/CCA/data/smoothed/HI2/'

IndList=np.loadtxt('/scratch/dsw310/CCA/data/extras/IndList/IndList_haloMix.dat')
IndList=IndList.astype(int)
   
class SimuData(Dataset):
    def __init__(self,lIndex,hIndex,hod=0,aug=0,test=0):
        self.datafiles = []
        self.hod=hod
        self.aug=aug
        self.test=test
        for i in np.arange(lIndex,hIndex):
            self.datafiles+=[IndList[i]]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index],self.hod,self.aug,self.test)

    def __len__(self):
        return len(self.datafiles)
        
def get_mini_batch(ind,hod,aug,test):
    inp=np.load(inpDir+str(ind)+'.npy')
    if hod:
        inp[1]=inp[1]/5.
    else:
        inp=np.split(inp,2,axis=0)[1]
    if test:
        if hod:
            return torch.from_numpy(inp[0]).float(),torch.from_numpy(inp[1]).float(),ind
        else:
            return torch.from_numpy(inp).float(),ind
    else:
        out=np.load(outDir+str(ind)+'.npy')
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
        if hod:
            inp=np.split(inp,2,axis=0)
            return torch.from_numpy(inp[0].copy()).float(),torch.from_numpy(inp[1].copy()).float(),torch.from_numpy(out.copy()).float()
        else:
            return torch.from_numpy(inp.copy()).float(),torch.from_numpy(out.copy()).float()
        
