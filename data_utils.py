import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import h5py

inpDir='/scratch/dsw310/CCA/data/DM+halos/'
outDir='/scratch/dsw310/CCA/data/HI/'
    
class SimuData(Dataset):
    def __init__(self,lIndex,hIndex):
        self.datafiles = []
        for x in np.arange(lIndex,hIndex):
            #y = [str(x)+'.npy']
            self.datafiles+=[x]

    def __getitem__(self, index):
        return get_mini_batch(self.datafiles[index])

    def __len__(self):
        return len(self.datafiles)
        
def get_mini_batch(ind):
    inp=np.load(inpDir+str(ind)+'.npy')
    out=np.load(outDir+str(ind)+'.npy')
    inp[1]=inp[1]/5.
    return torch.from_numpy(inp).float(),torch.from_numpy(out).float()
        
