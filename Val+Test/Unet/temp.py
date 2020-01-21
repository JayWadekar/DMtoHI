#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=16
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutTemp.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrTemp.log"

import numpy as np
import Pk_library as PKL
import void_library as VL
import time
start_time = time.time()

side_half=20; key='Ill_41'

BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0

delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy')
delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy')

threshold  = -0.7
#Radii      = np.array([1, 2.5], dtype=np.float32) #Mpc/h	    
Radii      = np.array([0.5, 1, 2, 4, 8], dtype=np.float32) #Mpc/h   
threads1   = 15
threads2   = 4


V = VL.void_finder(delta, BoxSize, threshold, Radii, threads1, threads2, void_field=False)
temp=np.hstack((np.reshape(V.Rbins ,(-1,1)),np.reshape(V.void_vsf ,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Void'+key+'.dat',temp)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

'''
# Bispectrum (Not working)
k1=0.5; k2=0.6
theta = np.linspace(0., np.pi, 5)

Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=5)
temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+'.dat',temp)
'''







