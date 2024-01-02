import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import sys

def getfile(f1):
  v=np.fromfile(f1,dtype=np.float32)
  v=v.reshape(nx,ny,nz)
  return v

file1=sys.argv[1]
file2=sys.argv[2]
nx=int(sys.argv[3])
ny=int(sys.argv[4])
nz=int(sys.argv[5])


I1=getfile(file1)
I2=getfile(file2)

scale=1


d1=I1[:,int(ny/2),:]
d2=I2[:,int(ny/2),:]
#plt.plot(d1[int(nx/2),:].T)
#plt.plot(d2[int(nx/2),:].T)

#plt.subplot(1,3,1)
d1=I1[:,int(ny/2),:]
plt.imshow(d1.T,aspect='auto',vmax=np.max(d1)*scale,vmin=np.min(d1)*scale)
plt.colorbar()
#plt.subplot(1,3,2)
#d2=I[int(nx/2),:,:]
#plt.imshow(d2.T,aspect='auto',vmax=np.max(d2)*scale,vmin=np.min(d2)*scale)
#plt.colorbar()
#plt.subplot(1,3,3)
#d3=I[:,:,2]
#plt.imshow(d3,aspect='auto',vmax=np.max(d1)*scale,vmin=np.min(d1)*scale)
#plt.colorbar()
plt.show()

