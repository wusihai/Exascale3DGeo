#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
import sys 
from scipy.ndimage import zoom

def getfile(f1):
  v=np.fromfile(f1,dtype=np.float32)
  v=v.reshape(nx,ny,nz)
  return v

nx=int(sys.argv[1])
ny=int(sys.argv[2])
nz=int(sys.argv[3])
file=(sys.argv[4])
file1=(sys.argv[5])
v=getfile(file)

v=v[150:650,150:650,:]
#v=zoom(v,(0.6,0.1,2))
#v=v*0.2
print(np.shape(v))

v.tofile(file1)

