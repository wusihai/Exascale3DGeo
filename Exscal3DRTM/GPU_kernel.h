#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#ifndef M
#define M 5
#endif

__global__ void extractRecordVSP(int nx, int ny, int nz, int pml, float *u2,  int gz, float *record);

__global__ void wavefield_update_tti(float *d_c, float *d_c2, float *d_pmlx, float *d_pmly, float *d_pmlz, 
	float *d_epsilon,float *d_delta,float *d_vp, float *d_the, float *d_phi, float dx, float dy, float dz,  float dt, float wavelet,
	int nx, int ny, int nz, int pml,int sx, int sy, int sz,float *ux, float *uy, float *uz,float *u1,  
	float *u2, float *S);

__global__ void wavefield_update_vti(float *d_c, float *d_c2, float *d_pmlx, float *d_pmly, float *d_pmlz, 
	float *d_epsilon,float *d_delta,float *d_vp, float dx, float dy, float dz,  float dt, float wavelet,
	int nx, int ny, int nz, int pml,int sx, int sy, int sz,float *ux, float *uy, float *uz,float *u1,  
	float *u2, float *S);

__global__ void exchange(int nx, int ny, int nz, int pml, float *u1, float *u2 );

__global__ void addsource(float *d_source, float wavelet, float *u3, int nx, int ny, int nz);
__global__ void addSourcePML(float *u3, float *d_ul, float *d_ur,float *d_ut,float *d_ub, float *d_uf,float *d_uba, int nx, int ny, int nz, int pml);

__global__ void wavefield_output_pml(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml);

__global__ void wavefield_output_pmlx(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml);

__global__ void wavefield_output_pmly(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml);

__global__ void wavefield_output_pmlz(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml);




//__global__ void loadData(float *u3, float *d_record, int nx, int ny, int nz, int pml);
__global__ void loadData( int nx, int ny, int nz, int pml,int gz,float *u3, float *d_record);

__global__ void Imaging(float *u_forward, float *u_inverse, float *d_image, float *d_light, int nx, int ny, int nz, int pml);

void checkCUDAerror(const char *msg);
void CHECK(hipError_t a);
__global__ void VTI_forward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void TTI_forward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void VTI_backward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void TTI_backward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void VTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void TTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);

#endif
