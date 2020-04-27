#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --mem=30GB
#SBATCH --cpus-per-task=2
#SBATCH -o "/scratch/dsw310/CCA/Val+Test/Unet/IO/OutHist.log"
#SBATCH -e "/scratch/dsw310/CCA/Val+Test/Unet/IO/ErrHist.log"
#SBATCH --mail-type=NONE
#SBATCH --mail-user=dsw310@nyu.edu

#Note: 
#srun -c2 -t0:02:00 --mem=15GB --pty /bin/bash

import numpy as np
import Pk_library as PKL
import time
start_time = time.time()

#------------ Making FullBox----------------------------------
side_half=20
cut=42-side_half
key='Unet' #key='try'


boxes=np.load('/scratch/dsw310/CCA/data/output/HIboxes.npy',allow_pickle=True)
delta=np.zeros(((side_half*2+1)*8,(side_half*2+1)*8,(side_half*2+1)*8)).astype(np.float32)
#hiMax = np.log10(2.+9658.145)

for i in range(len(boxes)):
    ind=boxes[i,0]
    ind2=(ind // 3969 - cut)*8; ind1= ((ind%3969) // 63 -cut)*8; ind0=(ind % 63 -cut)*8
    delta[ind2:ind2+8,ind1:ind1+8,ind0:ind0+8]=boxes[i,1]


delta=np.power(delta*2.,5.)-1.
#a=np.power(a+1.,0.2)/2. Inverse Transformation
np.save('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy',delta)
#delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/BoxUnet.npy')

#Power spectrum
BoxSize = 1.171875*(2*side_half+1); MAS  = 'CIC'; axis = 0
Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads=2)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.Pk[:,0],(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Pow'+key+'.dat',temp)

#Cross power
delta2=np.load('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_512Res.npy')[22*8+4:63*8+4,22*8+4:63*8+4,22*8+4:63*8+4]
Pk = PKL.XPk([delta2,delta], BoxSize, axis, MAS=['CIC','CIC'], threads=2)
temp=np.hstack((np.reshape(Pk.k3D,(-1,1)),np.reshape(Pk.XPk[:,0,0],(-1,1)),np.reshape(Pk.Pk[:,0,0],(-1,1)),np.reshape(Pk.Pk[:,0,1],(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/PowX'+key+'.dat',temp)

#------------ Histogram and 2D plot----------------------------------

delta= np.log10(2.+delta)
rbins = np.arange(-.1,4.,0.05)
centers=0.5*(rbins[1:]+rbins[:-1])
d1,d2=np.histogram(delta, bins=rbins)
temp2=np.hstack((centers.reshape(-1,1),d1.reshape(-1,1)))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Hist'+key+'.dat',temp2)

#hiTwo = np.mean(delta, axis=0)
#np.save('/scratch/dsw310/CCA/Val+Test/Unet/Figs/2D'+key+'.npy', hiTwo)


time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



