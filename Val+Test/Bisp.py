#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=01:35:00
#SBATCH --mem=95GB
#SBATCH --cpus-per-task=8
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutTemp.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrTemp.log"

#srun -c10 -t0:45:00 --mem=90GB --pty /bin/bash
import numpy as np
import Pk_library as PKL
import time, h5py
start_time = time.time()

side_half=20;
BoxSize = 1.171875*(2*side_half+1)#*1200./1312.;
MAS  = 'CIC'; axis = 0
k1=0.3; k2=0.6; key2='_'+str(k1)+'_'+str(k2)
theta = np.linspace(0., 3.1, 20) #np.pi


def run(delta):
    Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=6) #delta=delta[:1200,:1200,:1200]
    temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
    np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+key2+'.dat',temp)

key='Ill'
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxIll_41.npy'); run(delta)

key='Unet'
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxUnet_1127.npy'); run(delta)

key='HOD'
f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta=f['delta_HI'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]; f.close(); run(delta)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))





