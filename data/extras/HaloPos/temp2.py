#!/usr/bin/env python


#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
#SBATCH --mem=249GB
#SBATCH -o "/scratch/dsw310/CCA/Extras/HaloPos/output2.log"
#SBATCH -e "/scratch/dsw310/CCA/Extras/HaloPos/error2.log"

#249GB 6 tasks 5 threads [0:]
import numpy as np
from mpi4py import MPI
import smoothing_library as SL
import Pk_library as PKL
import units_library as UL
import MAS_library as MASL
import h5py,time

start_time = time.time()

rho_crit = UL.units().rho_crit

BoxSize = 75.0 #Mpc/h
R       = 0.3  #Mpc/h
dims = 2048; z=1
Filter  = 'Top-Hat'
threads = 5
MAS  = 'CIC'
axis = 0


M_HI=np.load('M_HI.npy')
halo_pos=np.load('halo_pos.npy')
delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)
MASL.MA(halo_pos, delta_HI, BoxSize, MAS, W=M_HI)

print ('HI max: {}'.format(np.amax(delta_HI)))


grid    = delta_HI.shape[0]
W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
delta_HI = SL.field_smoothing(delta_HI, W_k, threads)
delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

print ('done!')

Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 6)
np.savetxt('Pk_HOD_WithProfile_Smooth.dat', np.transpose([Pk.k3D, Pk.Pk[:,0]]))

g = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'w')
g.create_dataset('delta_HI',    data=delta_HI)
g.close()

print ('HI smooth max: {}'.format(np.amax(delta_HI)))


time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))





