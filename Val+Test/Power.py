#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=4
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutPow.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrPow.log"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dsw310@nyu.edu

#srun -c10 -t0:20:00 --mem=30GB --pty /bin/bash

import numpy as np
import Pk_library as PKL

side_half=20
key='Unet' #key='try'

BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0

delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/Box'+key+'.npy')
Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads=10)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.Pk[:,0],(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Pow'+key+'.dat',temp)
#np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/PowHOD_41.dat',temp)

#-------------------------------------------
# Bispectrum
k1=0.3; k2=0.6
theta = np.linspace(0, np.pi, 6)

Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=10)
temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+'.dat',temp)

#-------------------------------------------
# Voids (same as Void.py)

import void_library as VL
#srun -c20 -t0:30:00 --mem=30GB --pty /bin/bash
threshold  = -0.7
#Radii      = np.array([0.5, 1, 2, 4, 8], dtype=np.float32) #Mpc/h       
Radii      = np.array([0.5, 1, 2, 3, 4, 5, 6.5, 9], dtype=np.float32) #Mpc/h     

# identify voids		
V = VL.void_finder(delta, BoxSize, threshold, Radii, threads1, threads2, void_field=False)
temp=np.hstack((np.reshape(V.Rbins ,(-1,1)),np.reshape(V.void_vsf ,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Void/Void'+key+'.dat',temp)
#np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/VoidUnet.dat',temp)


#if void_field:  void_field  = V.void_field
#void_pos    = V.void_pos    #positions of the void centers
#void_radius = V.void_radius #radius of the voids
#-------------------------------------------
# Cross power with DM

delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxIll_41.npy')
delta2=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxUnet_1127.npy')

Pk = PKL.XPk([delta,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads=10)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.XPk[:,0,0],(-1,1)),np.reshape(Pk.Pk[:,0,0],(-1,1)),np.reshape(Pk.Pk[:,0,1],(-1,1))))
key='HOD'
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/PowX'+key+'.dat',temp)
#-------------------------------------------
#print ('Side Length: {}'.format((2*side_half+1)))

#plt.xlabel('k (h Mpc$^{-1}$)',fontsize=14)
#plt.ylabel("$P(k) [(Mpc/h)^3]$",fontsize=14)
#plt.savefig('/scratch/dsw310/CCA/Val+Test/Unet/Power.pdf')

#f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
#delta=f['delta_HI'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]

'''
Trash----------------------


delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxUnet_1127.npy')
Pk = PKL.XPk([delta,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads=10)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.Pk[:,0,0],(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/PowX'+key+'2.dat',temp)
'''

