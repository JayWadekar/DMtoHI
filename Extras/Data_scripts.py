#!/usr/bin/env python

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --mem=99GB
#SBATCH -o "/scratch/dsw310/CCA/trash/output2.log"
#SBATCH -e "/scratch/dsw310/CCA/trash/error2.log"

#--------------------------------------------------------------------------------------------------------------------
#Used to generate overdensity from the Halo Catalog

import numpy as np
import sys,os,h5py
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL
import time

start_time = time.time()
rho_crit = UL.units().rho_crit

BoxSize = 75.0 #Mpc/h

dims = 512
MAS  = 'CIC'
axis = 0

z=1
alpha, M0, Mmin = 0.53, 1.5e10, 6e11

f = h5py.File('/scratch/dsw310/CCA/data/extras/Halo_catalogue_z=%s.hdf5'%z, 'r')
halo_pos=f['pos'][0:]
halo_mass=f['mass'][0:]
f.close()

M_HI = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**(0.35))

Omega_HI = np.sum(M_HI, dtype=np.float64)/(BoxSize**3*rho_crit)
print 'Omega_HI(z=%d) = %.3e'%(z,Omega_HI)

delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)
MASL.MA(halo_pos, delta_HI, BoxSize, MAS, W=M_HI)
delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 8)

np.savetxt('Pk_HI_Nbody_real_space_z=%.1f.dat'%z, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
g = h5py.File('Halo_delta_HI_z=%s.hdf5'%z, 'w')
g.create_dataset('delta_HI', data=delta_HI)
g.close()

#--------------------------------------------------------------------------------------------------------------------
# Used to generate the I/O files
# Takes nearly 30min #srun -c15 -t0:30:00 --mem=70GB --pty /bin/bash
import numpy as np
import h5py, time, multiprocessing

#Max DM voxel is 459143940000.0
f = h5py.File('/scratch/dsw310/CCA/data/fields_z=1.0.hdf5', 'r'); a=f['delta_m'][0:];
a-=np.amin(a)
a=np.power(a,0.14) #0.1 and /5. was for FP
a/=15.
##a=np.power(a*5.,10.) #Inverse transform not complete


#Max smoothed HI voxel is 9659.14446446792
a = np.load('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_512Res.npy'); 
a-=np.amin(a)
a=np.power(a,0.28) #0.2 and /2. was for FP
a/=5.2
# mean=0.28818035, std = 0.3
#a=np.power(a*2.,5.)-1. #Inverse transform not complete
np.save('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_512Res_rescaled2.npy',a)

def box(ind):
    ind2=ind // 3969; ind1= (ind%3969) // 63; ind0=ind % 63
    
    dm=a[ind2*32:ind2*32+64,ind1*32:ind1*32+64,ind0*32:ind0*32+64]
    #hi=a[ind2*8+4:ind2*8+12,ind1*8+4:ind1*8+12,ind0*8+4:ind0*8+12]
    
    dm=np.expand_dims(dm,axis=0)
    #hi=np.expand_dims(hi,axis=0)
    
    np.save('/scratch/dsw310/CCA/data/DM_training2/'+str(ind)+'.npy',dm)   
    #np.save('/scratch/dsw310/CCA/data/smoothed/HI_training/'+str(ind)+'.npy',hi)
    return


start_time = time.time()
j=np.arange(0,pow(63,3),1)
p = multiprocessing.Pool(processes=15)
p.map(box, j)
p.close(); p.join()
print (time.time() - start_time)

#--------------------------------------------------------------------------------------------------------------------
#Determine threshold effect on power

BoxSize = 37.5 #Mpc/h
dims = 1024
delta_HI=f['delta_HI'][512:1536,512:1536,512:1536]

deltaMax=14264290.0; deltaMax = np.log10(2.+deltaMax)
thres=0.6
thres=np.power(10.,thres*deltaMax)-2.
mask_value = delta_HI > thres
mask_value =mask_value.astype(np.float32)
delta_HI = mask_value * delta_HI

Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 8)

#--------------------------------------------------------------------------------------------------------------------
#    HI smoothing

#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --mem=99GB
#SBATCH -o "/scratch/dsw310/CCA/trash/TrashOut/output2.log"
#SBATCH -e "/scratch/dsw310/CCA/trash/TrashOut/error2.log"

#249GB 6 tasks 5 threads [0:]
import numpy as np
from mpi4py import MPI
import smoothing_library as SL
import Pk_library as PKL
import h5py

f = h5py.File('/scratch/dsw310/CCA/data/fields_z=1.0.hdf5', 'r')
delta_HI=f['delta_HI'][0:]
f.close()

BoxSize = 75.0 #Mpc/h
R       = 0.3  #Mpc/h
grid    = delta_HI.shape[0]
Filter  = 'Top-Hat'
threads = 5
MAS  = 'CIC'
axis = 0

W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
delta_HI = SL.field_smoothing(delta_HI, W_k, threads)
delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

#Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, 8)
#np.savetxt('Pk_HI_smoothed=0.1.dat', np.transpose([Pk.k3D, Pk.Pk[:,0]]))

g = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_smoothed.hdf5', 'w')
g.create_dataset('delta_HI',    data=delta_HI)
g.close()

print ('HI max: {}'.format(np.amax(delta_HI)))


#---------------------------------------------------------------------------------------------------
# HOD halo profile make

#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --mem=99GB

import numpy as np
import sys,os,h5py,time
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL

start_time = time.time(); rho_crit = UL.units().rho_crit

n_particles=100
epsilon=1e-3

BoxSize = 75.0 #Mpc/h
dims = 2048;MAS  = 'CIC';axis = 0; z=1
alpha, M0, Mmin = 0.53, 1.5e10, 6e11


f = h5py.File('Halo_catalogue_z=%s.hdf5'%z, 'r')
halo_pos=f['pos'][0:]; halo_mass=f['mass'][0:]; halo_radius=f['radius'][0:]; num_halos = len(halo_pos)
f.close()

halo_mass = M0*(halo_mass/Mmin)**alpha*np.exp(-(Mmin/halo_mass)**(0.35))
halo_mass=np.multiply(np.ones((num_halos, n_particles)),halo_mass.reshape(-1,1)/n_particles).astype(np.float32).flatten()

r = np.random.rand(num_halos, n_particles)
phi = np.random.rand(num_halos, n_particles)
theta = np.random.rand(num_halos, n_particles)

r = epsilon*np.power(halo_radius.reshape(-1,1)/epsilon, r)
phi = 2*np.pi*phi
theta = np.arccos(-1+2*theta)

x = np.array(halo_pos[:,0]).reshape(-1,1) + np.multiply(np.multiply(r, np.sin(theta)), np.cos(phi))
y = np.array(halo_pos[:,1]).reshape(-1,1) + np.multiply(np.multiply(r, np.sin(theta)), np.sin(phi))
r = np.array(halo_pos[:,2]).reshape(-1,1) + np.multiply(r, np.cos(theta))
halo_pos = np.stack((x,y,r), axis=2)
halo_pos =halo_pos.reshape(num_halos*n_particles,3).astype(np.float32)

mask = (halo_pos>=0)&(halo_pos<=75); mask = mask.all(1); halo_pos = halo_pos[mask]; halo_mass = halo_mass[mask]

Omega_HI = np.sum(halo_mass, dtype=np.float64)/(BoxSize**3*rho_crit)
print 'Omega_HI(z=%d) = %.3e'%(z,Omega_HI)

#np.save('M_HI.npy',halo_mass)
#np.save('halo_pos.npy',halo_pos)

#M_HI=np.load('M_HI_100.npy')
#halo_pos=np.load('halo_pos_100.npy')


delta_HI = np.zeros((dims,dims,dims), dtype=np.float32)
MASL.MA(halo_pos, delta_HI, BoxSize, MAS, W=M_HI)
delta_HI /= np.mean(delta_HI, dtype=np.float64);  delta_HI -= 1.0

Pk = PKL.Pk(delta_HI, BoxSize, axis, MAS, threads=8)

np.savetxt('Pk_HOD_WithProfile.dat', np.transpose([Pk.k3D, Pk.Pk[:,0]]))
g = h5py.File('/scratch/dsw310/CCA/data/Halo_delta_HI_withProfile_z=%s.hdf5'%z, 'w')
g.create_dataset('delta_HI', data=delta_HI)
g.close()

#-----------------------------------------------------------------------
# HI reduce grid size

import numpy as np
import sys,os,h5py
import MAS_library as MASL
import Pk_library as PKL
import units_library as UL
import time
start_time = time.time()
rho_crit = UL.units().rho_crit
MAS  = 'CIC'
axis = 0

#BoxSize = 75.0 #Mpc/h

f = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_2048Res.hdf5', 'r')
a=f['delta_HI'][0:]; a+=1.
highres=2048

for i in range(1,highres-2,2):
    a[i-1,:,:]+=a[i,:,:]/2.; a[i+1,:,:]+=a[i,:,:]/2.; a[i,:,:]=0
    a[:,i-1,:]+=a[:,i,:]/2.; a[:,i+1,:]+=a[:,i,:]/2.; a[:,i,:]=0
    a[:,:,i-1]+=a[:,:,i]/2.; a[:,:,i+1]+=a[:,:,i]/2.; a[:,:,i]=0
    
a[highres-2,:,:]+=a[highres-1,:,:]/2.; a[0,:,:]+=a[highres-1,:,:]/2.; a[:,highres-2,:]+=a[:,highres-1,:]/2.
a[:,0,:]+=a[:,highres-1,:]/2.; a[:,:,highres-2]+=a[:,:,highres-1]/2.
a[:,:,0]+=a[:,:,highres-1]/2.; a[highres-1,:,:]=0.; a[:,highres-1,:]=0.; a[:,:,highres-1]=0.

b=a[::2,::2,::2]
b/=8.
b-=1.
g = h5py.File('/scratch/dsw310/CCA/data/smoothed/HI_smoothed_1024Res.hdf5', 'w')
g.create_dataset('delta_HI', data=b)
g.close()

BoxSize = 75.0*2047/2048.
#BoxSize = 75.0*2047/2048.*1023./1024. #For 512 box

Pk = PKL.Pk(b, BoxSize, axis, MAS, 2)

np.savetxt('/scratch/dsw310/CCA/data/extras/Pk_HOD_smooth_1024Res.dat', np.transpose([Pk.k3D, Pk.Pk[:,0]]))

#--------------------------------------------------------------------------------------------------------------------
#Used to generate npy I/O files (Older version with full Res HI)

f = h5py.File('/scratch/dsw310/CCA/data/fields_z=1.0.hdf5', 'r')
#dmMax=5.037600040435791 #DM max
dmMax=6.114471435546875 #HI max    
        
#g = h5py.File('/scratch/dsw310/CCA/data/Halo_delta_HI_z=1.hdf5', 'r')
haloMax=14264290.0
haloMax = np.log10(2.+haloMax)

for ind in range(0,pow(63,3)):#
    print (ind)
    ind2=ind // 3969; ind1= (ind%3969) // 63; ind0=ind % 63
    #halo=g['delta_HI'][ind2*32:ind2*32+64,ind1*32:ind1*32+64,ind0*32:ind0*32+64]
    #dm=f['delta_m'][ind2*32:ind2*32+64,ind1*32:ind1*32+64,ind0*32:ind0*32+64]
    dm=f['delta_HI'][ind2*32+16:ind2*32+48,ind1*32+16:ind1*32+48,ind0*32+16:ind0*32+48]
    dm=dm/dmMax
    #halo=np.log10(2.+halo)
    #halo=halo/haloMax
    #halo=np.expand_dims(halo,axis=0)
    dm=np.expand_dims(dm,axis=0)
    #np.save('/scratch/dsw310/CCA/data/DM+halos/'+str(ind)+'.npy',np.concatenate((halo, dm), axis=0))
    np.save('/scratch/dsw310/CCA/data/HI/'+str(ind)+'.npy',dm)     
         
f.close()
#g.close()

#--------------------------------------------------------------------------------------------------
#Galaxy field from TNG100-1 directly from Illustris website

import illustris_python as il
import MAS_library as MASL
import smoothing_library as SL

basePath = '/scratch/dsw310/CCA/TNG/data/TNG100-1/z_1.0'
fields = ['SubhaloMass','SubhaloPos','SubhaloFlag'] #SubhaloCM
subhalos = il.groupcat.loadSubhalos(basePath,50,fields=fields)

mask= subhalos['SubhaloFlag'] ==1
mass= subhalos['SubhaloMass'][mask]/ 0.704
pos= (subhalos['SubhaloPos'][mask])/1000.

mask=mass>=.1
mass= mass[mask]; pos= pos[mask]

BoxSize = 75. #Mpc/h
dims = 512; MAS  = 'CIC';axis = 0;

delta_m = np.zeros((dims,dims,dims), dtype=np.float32)
MASL.MA(pos, delta_m, BoxSize, MAS, W=np.ones(len(pos),dtype=np.float32));
#np.save('/scratch/dsw310/CCA/data/Galaxies2.npy',delta_m)

R       = 0.3; grid    = delta_m.shape[0]
Filter  = 'Top-Hat'; threads = 2; axis = 0

W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
delta_m = SL.field_smoothing(delta_m, W_k, threads)
delta_m /= np.mean(delta_m, dtype=np.float64);  delta_m -= 1.0
#np.save('/scratch/dsw310/CCA/data/Galaxies2.npy',delta_m)



#--------------------------------------------------------------------------------------------------