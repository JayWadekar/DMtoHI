#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=2
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutHist.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrHist.log"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dsw310@nyu.edu


#srun -c4 -t0:20:00 --mem=10GB --pty /bin/bash
#srun -c2 -t1:30:00 --mem=40GB --pty /bin/bash

import numpy as np
import Pk_library as PKL
import time
start_time = time.time()

#------------ Making FullBox----------------------------------
side_half=20
dire1='/scratch/dsw310/CCA/data/output/HIboxes/'
#dire2='/scratch/dsw310/CCA/data/smoothed/HI/'
key='Unet' #key='try'

start=42*3969+42*63+42 #32146# 19716 
delta=np.zeros(((side_half*2+1)*32,(side_half*2+1)*32,(side_half*2+1)*32)).astype(np.float32)
hiMax = np.log10(2.+10541.7539062)

for i in range(-side_half,side_half+1):
    for j in range(-side_half,side_half+1):
        for k in range(-side_half,side_half+1):
            ind=start+i*3969+j*63+k
            mx=(i+side_half)*32; my=(j+side_half)*32; mz=(k+side_half)*32; 
            delta[mx:mx+32,my:my+32,mz:mz+32]=np.load(dire1+str(ind)+'.npy')[0,0] #37minutes


#------------ Histogram and 2D plot----------------------------------

rbins = np.arange(-.201,1.02,0.01)
centers=0.5*(rbins[1:]+rbins[:-1])


d1,d2=np.histogram(delta, bins=rbins)
temp2=np.hstack((centers.reshape(-1,1),d1.reshape(-1,1)))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Hist'+key+'.dat',temp2)


#hiTwo = np.mean(delta, axis=0)
#np.save('/scratch/dsw310/CCA/Val+Test/Unet/Figs/2D'+key+'.npy', hiTwo)

delta = np.power(10.,delta*hiMax)-2.
np.save('/scratch/dsw310/CCA/Val+Test/Unet/Box/Box'+key+'.npy',delta)

BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0

Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads=2)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.Pk[:,0],(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Pow'+key+'.dat',temp)
#np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/PowHOD_41.dat',temp)

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



