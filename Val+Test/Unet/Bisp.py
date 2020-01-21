#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutTemp.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrTemp.log"

#Does not work currently
import numpy as np
import Pk_library as PKL
import time
start_time = time.time()

side_half=20; key='Ill_41'

BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0

delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy')

k1=0.5; k2=0.6
theta = np.linspace(0., np.pi, 5)

Bk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS='CIC', threads=5)
temp=np.hstack((np.reshape(theta,(-1,1)),np.reshape(Bk.B,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Bisp'+key+'.dat',temp)


time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))





