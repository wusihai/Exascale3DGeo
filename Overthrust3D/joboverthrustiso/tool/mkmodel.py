
import numpy as np
#from matplotlib import pyplot as plt
#from scipy import ndimage
import sys 
nx=200
ny=200
nz=200
model=np.ones((nx,ny,nz),dtype='float32')
model=model*1800;
model[:,:,0:int(nz/4)]=1200;
model[:,:,int(nz/4):int(nz/2)]=1400;
model[:,:,int(nz/2):int(3*nz/4)]=1600;
model.tofile("./model/model.dat_1")
#plt.imshow(model.T)
#plt.show()
