#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
import sys 

nx=int(sys.argv[1])
ny=int(sys.argv[2])
nz=int(sys.argv[3])
it=int((sys.argv[4]));
#f=open("./snap/forward.0.0_%d.dat"%(it))
#f=open("./snap/image_%d.dat"%(it))
f=open("./snap/Total_image.0.0.dat")
#f=open("./snap/backward_%d.dat"%(it))
#f=open("./snap/extrap_%d.dat"%(it))
#f=open("./snap/recordVz.0.0.dat")
snap=np.fromfile(f,dtype='float32');
snap=snap.reshape(nx,ny,nz)

print(np.shape(snap))
snap[nx/2,:,:].tofile("snap_y%d_z%d.dat"%(ny,nz))
#plt.imshow(snap[nx/2,40:-40,40:-40].T)
plt.imshow(snap[nx/2,:,:].T)
plt.show()
plt.colorbar()

