
EXE=../bin/forward3d
CPP=mpic++
CFLAGS=-O3 -fopenmp -c 

GPUCFLAGS= -w -c -O3 -DTCE_HIP -fno-gpu-rdc -std=c++11  --amdgpu-target=gfx906  
IDIR=-I/public/home/tju_wusihai/soft/fftw/include -I/public/home/tju_wusihai/soft/gptl/include  -I/public/home/tju_wusihai/soft/zfp-0.5.5/include/

LFLAGS= -fopenmp -O3 -w -std=c++11
LIBS=-L/opt/rocm/lib/    -lhip_hcc -L/opt/rocm/lib/ -lhip_hcc -L/public/home/tju_wusihai/soft/fftw/lib/  -lfftw3f  -lfftw3f_omp -L/public/home/tju_wusihai/soft/gptl/lib -lgptl -lpthread -L/public/home/tju_wusihai/soft/zfp-0.5.5/build/lib64 -lzfp  -L/public/home/tju_wusihai/soft/zfp-0.5.5/build -lcuda_zfp 

$(EXE) :  allocate.o assistant.o initialization.o main.o rtmlib.o GPU_kernel.o  Propagator.o  #librtm3d.so 
	mpic++ ${LFLAGS} -o $@ $^ ${LIBS} 
Propagator.o : Propagator.cpp
	hipcc ${GPUCFLAGS} ${IDIR}  -o $@ $< 
GPU_kernel.o : GPU_kernel.cpp
	hipcc ${GPUCFLAGS}  ${IDIR}  -o $@ $<
.cpp.o:
	mpic++ ${CFLAGS} ${IDIR}  $<

clean:
	rm *.so *.o -rf
install :
	cp $(EXE) ../job
run :
	mpirun -np $(NP) ./3dqpmodeling 
