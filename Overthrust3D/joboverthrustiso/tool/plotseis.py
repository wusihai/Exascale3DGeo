
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.ticker import FuncFormatter
import sys 
def changex(temp, position):
        return int(temp/4)

def changey(temp, position):
            return int(temp/100)

nx=int(sys.argv[1])
ny=int(sys.argv[2])
nz=int(sys.argv[3])
subx=int(nx/int(sys.argv[4]))
suby=int(ny/int(sys.argv[5]))
record1=np.zeros((nx,ny,nz),dtype='float32')
for i in range(int(sys.argv[4])):
  for j in range(int(sys.argv[5])):
    print("./record/record%d-%d.dat"%(j,i))
    f=open("./record/record%d-%d.dat"%(j,i))
    record=np.fromfile(f,dtype='float32');
    record1[0+subx*i:subx+subx*i,0+suby*j:suby+suby*j,:]=record.reshape(subx,suby,nz)

record1.tofile("./record/recordrtm.dat")
ax=plt.gca()
plt.subplot(121)
plt.imshow(record1[100,30:-30,:].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(record1[30:-30,100,1000:].T)
plt.colorbar()
#plt.subplot(133)
#plt.imshow(record1[30:-30,30:-30,400].T)
ax.invert_yaxis()
plt.show()
