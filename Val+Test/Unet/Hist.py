

#srun -c4 -t0:20:00 --mem=10GB --pty /bin/bash
#srun -c2 -t1:30:00 --mem=40GB --pty /bin/bash
import numpy as np

#------------ Making FullBox----------------------------------
side_half=20
dire1='/scratch/dsw310/CCA/data/output/HIboxes2/'
dire2='/scratch/dsw310/CCA/data/smoothed/HI/'
key='Unet_41'

start=42*3969+42*63+42 #32146# 19716 
delta=np.zeros(((side_half*2+1)*32,(side_half*2+1)*32,(side_half*2+1)*32)).astype(np.float32)
hiMax = np.log10(2.+10541.7539062)

for i in range(-side_half,side_half+1):
    for j in range(-side_half,side_half+1):
        for k in range(-side_half,side_half+1):
            ind=start+i*3969+j*63+k
            mx=(i+side_half)*32; my=(j+side_half)*32; mz=(k+side_half)*32; 
            delta[mx:mx+32,my:my+32,mz:mz+32]=np.load(dire1+str(ind)+'.npy')[0,0]



#------------ Histogram and 2D plot----------------------------------

rbins = np.arange(-.201,1.02,0.01)
centers=0.5*(rbins[1:]+rbins[:-1])

#delta=np.load('/scratch/dsw310/CCA/Val+Test/Unet/BoxUnet.npy')
d1,d2=np.histogram(delta, bins=rbins)
temp2=np.hstack((np.reshape(centers,(-1,1)),np.reshape(d1,(-1,1))))
np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/Hist'+key+'.dat',temp2)
#np.savetxt('/scratch/dsw310/CCA/Val+Test/Unet/Figs/HistUnet_1123.dat',temp2)


#hiTwo = np.mean(delta, axis=0)
#np.save('/scratch/dsw310/CCA/Val+Test/Unet/Figs/2D'+key+'.npy', hiTwo)


delta = np.power(10.,delta*hiMax)-2.
np.save('/scratch/dsw310/CCA/Val+Test/Unet/Box'+key+'.npy',delta)

#------------------------------------
# For HOD
import h5py
f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_HOD_WithProfile_smoothed.hdf5', 'r')
delta=f['delta_HI'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]
delta=f['delta_HI'][32*32+16:54*32-16,32*32+16:54*32-16,32*32+16:54*32-16]
f.close()

hiMax= np.log10(7374.85986328+2.)
delta=np.log10(delta+2.)/hiMax

#DM below, not used
#f = h5py.File('/scratch/dsw310/CCA/data/fields_z=1.0.hdf5', 'r')
#delta=f['delta_m'][22*32+16:64*32-16,22*32+16:64*32-16,22*32+16:64*32-16]; f.close()
#delta = np.power(10.,delta)-2.

