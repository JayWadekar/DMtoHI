#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=0:05:00
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=2
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutTemp.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrTemp.log"

#srun -c2 -t0:05:00 --mem=30GB --pty /bin/bash
import numpy as np
import Pk_library as PKL
import time, h5py
start_time = time.time()

side_half=20;
BoxSize = 1.171875*(2*side_half+1)#*1200./1312.;
MAS  = 'CIC'; axis = 0

k1=0.4; k2=0.5; key2='_'+str(k1)+'_'+str(k2)
theta = np.linspace(0., 3.1, 20) #np.pi


def run(delta):
    Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=2) #delta=delta[:1200,:1200,:1200]
    temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
    np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+key2+'.dat',temp)

key='Ill'
delta=np.load('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_512Res.npy')[22*8+4:63*8+4,22*8+4:63*8+4,22*8+4:63*8+4]; run(delta)

key='Unet'
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/BoxUnet.npy'); run(delta)

key='HOD'
delta=np.load('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed_512Res.npy')[22*8+4:63*8+4,22*8+4:63*8+4,22*8+4:63*8+4]; run(delta)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

#--------------------------------------------------
#Making equilateral triangles

theta = np.linspace(np.pi/3., 3.1, 1)
k1=np.exp(np.linspace(np.log(0.2),np.log(4.),20))
#k1=np.exp(np.linspace(np.log(2.),np.log(4.),3))

def run(delta):
    temp=np.zeros(len(k1))
    Bk = PKL.Bk(delta, BoxSize, k1[1], k1[1], theta, MAS='CIC', threads=2)
    for i in range(len(k1)):
        Bk = PKL.Bk(delta, BoxSize, k1[i], k1[i], theta, MAS='CIC', threads=2)
        temp[i]=Bk.B
    temp=np.hstack((np.reshape(k1,(-1,1)),np.reshape(temp,(-1,1))))
    np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+'.dat',temp)