#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
import sys 

nx=int(sys.argv[1])
ny=int(sys.argv[2])
nz=int(sys.argv[3])
file=(sys.argv[4])

v=np.ones((nx,ny,nz),dtype=np.float32)*0.1
#v=np.ones((nx,ny,nz),dtype=np.float32)*3000
#v[:,:,0:40]=2400;
#v[:,:,41:60]=2600;
v.tofile(file)

