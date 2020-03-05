#srun -c4 -t0:20:00 --mem=10GB --pty /bin/bash
#srun -c2 -t1:15:00 --mem=40GB --pty /bin/bash


import numpy as np

#------------------------------------
# For HOD
import h5py
f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta=f['delta_HI'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]
#delta=f['delta_HI'][32*32+16:54*32-16,32*32+16:54*32-16,32*32+16:54*32-16]
f.close()

hiMax= np.log10(7374.85986328+2.)
delta=np.log10(delta+2.)/hiMax

#------------------------------------
# 2D difference plots
delta2 = np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxIll_41.npy')
delta = np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box/BoxUnet_1127.npy')

hiMax = np.log10(2.+10541.7539062)
delta=np.log10(delta+2.)/hiMax

delta=np.absolute(delta-delta2)

hiTwo = np.mean(delta, axis=0)
np.save('/scratch/dsw310/CCA/Val+Test/Unet/Figs/2D_diff_Unet.npy', hiTwo)

np.save('/scratch/dsw310/CCA/Val+Test/Unet/Figs/2D_diff_HOD.npy', hiTwo)


#DM below, not used
#f = h5py.File('/scratch/dsw310/CCA/data/fields_z=1.0.hdf5', 'r')
#delta=f['delta_m'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]; f.close()
#delta = np.power(10.,delta)-2.







