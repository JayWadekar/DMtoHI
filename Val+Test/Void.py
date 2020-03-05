#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=20
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutVoid2.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrVoid2.log"


#Look at input files and their

import numpy as np
import Pk_library as PKL
import void_library as VL
import time, h5py
start_time = time.time()

void=1
bisp=0

side_half=10;
BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0

threshold  = -0.7
#Radii      = np.array([1, 2.5], dtype=np.float32) #Mpc/h	    
Radii      = np.array([0.1, 0.2, 0.4, 0.8, 1.5, 3., 4.5], dtype=np.float32) #Mpc/h   
threads1   = 20
threads2   = 4

k1=0.4; k2=0.6
theta = np.linspace(0, np.pi, 10)

def stats(void=0, bisp=0):
    if (void):
        V = VL.void_finder(delta, BoxSize, threshold, Radii, threads1, threads2, void_field=False)
        temp=np.hstack((np.reshape(V.Rbins ,(-1,1)),np.reshape(V.void_vsf ,(-1,1))))
        np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Void/Void'+key+'.dat',temp)
        
    if (bisp):
        Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=10)
        temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
        np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp/Bisp'+key+'.dat',temp)
'''
key='Ill_21'
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy')
stats(void=void, bisp=bisp)


key='HOD_21'
f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta=f['delta_HI'][32*32+16:54*32-16,32*32+16:54*32-16,32*32+16:54*32-16]
f.close()
stats(void=void, bisp=bisp)
'''

key='Unet_21'
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/BoxUnet_1127.npy') #*****LOOK
delta=delta[10*32+16:32*32-16,10*32+16:32*32-16,10*32+16:32*32-16] #*****LOOK
stats(void=void, bisp=bisp)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))








