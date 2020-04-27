#!/usr/bin/env python


#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:20:00
#SBATCH --mem=50GB
#SBATCH -o "/scratch/dsw310/CCA/Extras/HaloPos/output1.log"
#SBATCH -e "/scratch/dsw310/CCA/Extras/HaloPos/error1.log"

import numpy as np
import sys,os,h5py,time
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL

start_time = time.time(); rho_crit = UL.units().rho_crit

BoxSize = 75.0 #Mpc/h
R       = 0.3  #Mpc/h
MAS  = 'CIC'
axis = 0

f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta_HI=f['delta_HI'][0:]
f.close()

Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 10)
np.savetxt('Pk_HOD_WithProfile_Smooth.dat', np.transpose([Pk.k3D, Pk.Pk[:,0]]))

'''
#Comment before
f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta=f['delta_HI'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]
f.close()


Pk = PKL.Pk(delta, BoxSize, axis, MAS, 10)
'''





