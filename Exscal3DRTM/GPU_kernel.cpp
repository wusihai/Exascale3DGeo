#include "hip/hip_runtime.h"
// ********************************************************************** 
// ********************************************************************** 
//                      3D Acoustic Modeling (Pure P wave) 
//      (C) Copyright Tongji,  Author : Tengfei Wang, Lun Ruan and 
//		Jiubing Cheng, 2019 
//      The propagator is written by Lun Ruan, 2018.10 
//      originated from Xu,2015, modified by Tengfei Wang, 2019.08 
// 
// ********************************************************************** 
// ********************************************************************** 

#include <stdio.h>
#include <math.h>
#include "GPU_kernel.h"
#define EPS  1.e-20
//#define blocksizeMax 1024




__global__ void addSourcePML(float *u3, float *d_ul, float *d_ur,float *d_ut,float *d_ub, float *d_uf,float *d_uba, int nx, int ny, int nz, int pml)
{
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;


	if(idx<nx && idy<ny && idz<nz)
	{
		int i = idx*ny*nz+idy*nz+idz;


		if(idx>=pml-5 && idx<pml && idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
			u3[i]=d_ul[(idx-(pml-5))*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml];


		else if(idx>=nx-pml && idx<nx-pml+5 && idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
			u3[i]=	d_ur[(idx-(nx-pml))*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml] ;

		else if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=pml-5 && idz<pml)
			u3[i]=d_ut[(idz-(pml-5))*(nx-2*pml)*(ny-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml] ;

		else if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=nz-pml && idz<nz-pml+5)
			u3[i]=d_ub[(idz-(nz-pml))*(nx-2*pml)*(ny-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml];

		else if(idx>=pml && idx<nx-pml && idz>=pml && idz<nz-pml && idy>=pml-5 && idy<pml)
			u3[i]=d_uf[(idy-(pml-5))*(nx-2*pml)*(nz-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] ;

		else if(idx>=pml && idx<nx-pml && idz>=pml && idz<nz-pml && idy>=ny-pml && idy<ny-pml+5)
			u3[i]=d_uba[(idy-(ny-pml))*(nx-2*pml)*(nz-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] ;
	}
}

__global__ void wavefield_output_pml(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml)
{

	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;

	if(idx<nx && idy<ny && idz<nz)
	{
		int i = idx*ny*nz+idy*nz+idz;

		d_ut1[i] = u1[i];
		d_ut2[i] = u2[i];

		if(idx>=pml-5 && idx<pml && idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
			d_ul[(idx-(pml-5))*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml] = u2[i];
		else if(idx>=nx-pml && idx<nx-pml+5 && idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
			d_ur[(idx-(nx-pml))*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml] = u2[i];

		else if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=pml-5 && idz<pml)
			d_ut[(idz-(pml-5))*(nx-2*pml)*(ny-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml] = u2[i];

		else if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml && idz>=nz-pml && idz<nz-pml+5)
			d_ub[(idz-(nz-pml))*(nx-2*pml)*(ny-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml] = u2[i];

		else if(idx>=pml && idx<nx-pml && idz>=pml && idz<nz-pml && idy>=pml-5 && idy<pml)
			d_uf[(idy-(pml-5))*(nx-2*pml)*(nz-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] = u2[i];

		else if(idx>=pml && idx<nx-pml && idz>=pml && idz<nz-pml && idy>=ny-pml && idy<ny-pml+5)
			d_uba[(idy-(ny-pml))*(nx-2*pml)*(nz-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] = u2[i];
	}
}

__global__ void wavefield_output_pmlx(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml)
{

	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	for(int x=0;x<5;x++){
		if(idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
		{
			int i = (x+pml-5)*ny*nz+idy*nz+idz;
			d_ul[x*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml] = u2[i];
		}
		if(idy>=pml && idy<ny-pml && idz>=pml && idz<nz-pml)
		{
			int i = (nx-pml+x)*ny*nz+idy*nz+idz;
			d_ur[x*(ny-2*pml)*(nz-2*pml)+(idy-pml)*(nz-2*pml)+idz-pml] = u2[i];
		}
	}
}
__global__ void wavefield_output_pmly(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml)
{
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	for(int y=0;y<5;y++){
		if(idz>=pml && idz<nz-pml && idx>=pml && idx<nx-pml)
		{
			int i = idx*ny*nz+(y+pml-5)*nz+idz;
			d_uf[y*(nx-2*pml)*(nz-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] = u2[i];
		}
		if(idz>=pml && idz<nz-pml && idx>=pml && idx<nx-pml)
		{
			int i = idx*ny*nz+(y+ny-pml)*nz+idz;
			d_uba[y*(nz-2*pml)*(nx-2*pml)+(idx-pml)*(nz-2*pml)+idz-pml] = u2[i];
		}
	}
}
__global__ void wavefield_output_pmlz(float *u1, float *u2 ,float *d_ut1, float *d_ut2, float *d_ul, float *d_ur, float *d_ut,
		float *d_ub, float *d_uf, float *d_uba, int nx, int ny, int nz, int pml)
{
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;

	for(int z=0;z<5;z++){
		if(idy>=pml && idy<ny-pml && idx>=pml && idx<nx-pml)
		{
			int i = idx*ny*nz+idy*nz+z+pml-5;
			d_ut[z*(ny-2*pml)*(nx-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml] = u2[i];
		}
		if(idy>=pml && idy<ny-pml && idx>=pml && idx<nx-pml)
		{
			int i = idx*ny*nz+idy*nz+z+nz-pml;
			d_ub[z*(ny-2*pml)*(nx-2*pml)+(idx-pml)*(ny-2*pml)+idy-pml] = u2[i];
		}
	}
}

__global__ void loadData( int nx, int ny, int nz, int pml,int gz,float *u2, float *record)
{
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;

	if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml)
		u2[idx*ny*nz+idy*nz+gz]=record[(idx-pml)*(ny-2*pml)+idy-pml];
}
__global__ void Imaging(float *u_forward, float *u_inverse, float *d_image, float *d_light, int nx, int ny, int nz, int pml)
{
	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idx =  blockDim.z * blockIdx.z + threadIdx.z;

	int indexi = idx*(ny-2*pml)*(nz-2*pml)+idy*(nz-2*pml)+idz;
	int indexj = (idx+pml)*ny*nz+(idy+pml)*nz+idz+pml;

	if(idx<nx-2*pml && idz<nz-2*pml && idy<ny-2*pml){
		d_light[indexi] += u_forward[indexj]*u_forward[indexj];
		d_image[indexi] += u_forward[indexj]*u_inverse[indexj];
	}
}

#if 0
//Forward
__global__ void VTI_forward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idx =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	__shared__ float uS[18][18][18];
	//1.1 Load u1 to shared memory uS
	if(index>=0&&index<(nx*ny*nz))
		uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=u1[index];

	if(threadIdx.x<5){
		if((idx-5)*ny*nz+idy*nz+idz>=0&&(idx+8)*ny*nz+idy*nz+idz<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x    ]    =u1[(idx-5)*ny*nz+idy*nz+idz];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=u1[(idx+8)*ny*nz+idy*nz+idz];
		}
	}

	if(threadIdx.y<5){
		if(idx*ny*nz+(idy-5)*nz+idz>=0&&idx*ny*nz+(idy+8)*nz+idz<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y    ][threadIdx.x+5]=u1[idx*ny*nz+(idy-5)*nz+idz];
			uS[threadIdx.z+5][threadIdx.y+8+5][threadIdx.x+5]=u1[idx*ny*nz+(idy+8)*nz+idz];
		}
	}

	if(threadIdx.z<5){
		if(idx*ny*nz+idy*nz+idz-5>=0&&idx*ny*nz+idy*nz+idz+8<(nx*ny*nz)){
			uS[threadIdx.z]    [threadIdx.y+5][threadIdx.x+5]=u1[idx*ny*nz+idy*nz+idz-5];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=u1[idx*ny*nz+idy*nz+idz+8];
		}
	}
	__syncthreads();
	//1.2 Calculate the wavefield gradient  Ux,Uy,Uz,Uxx,Uyy,Uzz
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux+=d_c[M+m]*uS[threadIdx.x+5+m][threadIdx.y+5][threadIdx.z+5]/(2*dx);
			Uy+=d_c[M+m]*uS[threadIdx.x+5][threadIdx.y+5+m][threadIdx.z+5]/(2*dy) ;
			Uz+=d_c[M+m]*uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+5+m]/(2*dz); 
		}
		//calculate scalar_operator 

		S = 0.5*(1+sqrtf(1-8*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux+=d_c2[M+m]*uS[threadIdx.x+5+m][threadIdx.y+5][threadIdx.z+5]/(dx*dx);
			Uy+=d_c2[M+m]*uS[threadIdx.x+5][threadIdx.y+5+m][threadIdx.z+5]/(dy*dy) ;
			Uz+=d_c2[M+m]*uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+5+m]/(dz*dz); 
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient
		if(idx == sx && idy == sy && idz == sz){
			u2[index] = 2*u1[index]-u2[index]+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index] + wavelet;
		}
		else{
			u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
		}
	}
}

__global__ void VTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	__shared__ float uS[18][18][18];
	//1.1 Load u1 to shared memory uS
	if(index>=0&&index<(nx*ny*nz))
		uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+5]=u1[index];

	if(threadIdx.x<5){
		if((idx-5)*ny*nz+idy*nz+idz>=0&&(idx+8)*ny*nz+idy*nz+idz<(nx*ny*nz)){
			uS[threadIdx.x][threadIdx.y+5][threadIdx.z+5]    =u1[(idx-5)*ny*nz+idy*nz+idz];
			uS[threadIdx.x+8+5][threadIdx.y+5][threadIdx.z+5]=u1[(idx+8)*ny*nz+idy*nz+idz];
		}
	}

	if(threadIdx.y<5){
		if(idx*ny*nz+(idy-5)*nz+idz>=0&&idx*ny*nz+(idy+8)*nz+idz<(nx*ny*nz)){
			uS[threadIdx.x+5][threadIdx.y][threadIdx.z+5]    =u1[idx*ny*nz+(idy-5)*nz+idz];
			uS[threadIdx.x+5][threadIdx.y+8+5][threadIdx.z+5]=u1[idx*ny*nz+(idy+8)*nz+idz];
		}
	}

	if(threadIdx.z<5){
		if(idx*ny*nz+idy*nz+idz-5>=0&&idx*ny*nz+idy*nz+idz+8<(nx*ny*nz)){
			uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z]    =u1[idx*ny*nz+idy*nz+idz-5];
			uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+8+5]=u1[idx*ny*nz+idy*nz+idz+8];
		}
	}
	__syncthreads();
	//1.2 Calculate the wavefield gradient  Ux,Uy,Uz,Uxx,Uyy,Uzz
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux+=d_c[M+m]*uS[threadIdx.x+5+m][threadIdx.y+5][threadIdx.z+5]/(2*dx);
			Uy+=d_c[M+m]*uS[threadIdx.x+5][threadIdx.y+5+m][threadIdx.z+5]/(2*dy) ;
			Uz+=d_c[M+m]*uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+5+m]/(2*dz); 
		}
		//calculate scalar_operator 

		S = 0.5*(1+sqrtf(1-8*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux+=d_c2[M+m]*uS[threadIdx.x+5+m][threadIdx.y+5][threadIdx.z+5]/(dx*dx);
			Uy+=d_c2[M+m]*uS[threadIdx.x+5][threadIdx.y+5+m][threadIdx.z+5]/(dy*dy) ;
			Uz+=d_c2[M+m]*uS[threadIdx.x+5][threadIdx.y+5][threadIdx.z+5+m]/(dz*dz); 
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient
		if(idx == sx && idy == sy && idz == sz){
			u2[index] = 2*u1[index]-u2[index]+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index] + wavelet;
		}
		else{
			u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
		}
	}
}
#endif
#if 1
//Backward
__global__ void VTI_backward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){

	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idx =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		//calculate the wavefield gradient  Ux,Uy,Uz
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux += d_c[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(2*dx);
			Uy += d_c[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(2*dy); 
			Uz += d_c[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(2*dz); 
		}
		//calculate scalar_operator 
		S = 0.5*(1+sqrtf(1-8*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		//update wavefield (internal part)
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f; //!!!first derivative Ux respresent second derivative Uxx 
		for(m=-M;m<=M;m++){
			Ux += d_c2[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(dx*dx);
			Uy += d_c2[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(dy*dy); 
			Uz += d_c2[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(dz*dz);
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient

		u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
	}
}
//Forward
__global__ void VTI_forward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idx =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		//calculate the wavefield gradient  Ux,Uy,Uz
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux += d_c[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(2*dx);
			Uy += d_c[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(2*dy); 
			Uz += d_c[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(2*dz); 
		}
		//calculate scalar_operator 
		S = 0.5f*(1.0f+sqrtf(1.0f-8.0f*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		//update wavefield (internal part)
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f; //!!!first derivative Ux respresent second derivative Uxx 
		for(m=-M;m<=M;m++){
			Ux += d_c2[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(dx*dx);
			Uy += d_c2[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(dy*dy); 
			Uz += d_c2[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(dz*dz);
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient
		if(idx == sx && idy == sy && idz == sz){
			u2[index] = 2.0f*u1[index]-u2[index]+ dt*dt*((1.0f+2.0f*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index] + wavelet;
		}
		else{
			u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1.0f+2.0f*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
		}
	}
}
#endif 
//Reconstruction
__global__ void VTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){

	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idx =  blockDim.z * blockIdx.z + threadIdx.z;

	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
#if 1
		//calculate the wavefield gradient  Ux,Uy,Uz
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux += d_c[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(2*dx);
			Uy += d_c[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(2*dy); 
			Uz += d_c[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(2*dz); 
		}
		//calculate scalar_operator 
		S = 0.5*(1+sqrtf(1-8*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		//update wavefield (internal part)
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f; //!!!first derivative Ux respresent second derivative Uxx 
		for(m=-M;m<=M;m++){
			Ux += d_c2[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(dx*dx);
			Uy += d_c2[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(dy*dy); 
			Uz += d_c2[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(dz*dz);
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient

		u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
#endif
	}
}
#if 0
//Reconstruction
__global__ void VTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){

	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S;
	int m;
	int index=idx*ny*nz+idy*nz+idz;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		//calculate the wavefield gradient  Ux,Uy,Uz
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
		for(m=-M;m<=M;m++){
			Ux += d_c[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(2*dx);
			Uy += d_c[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(2*dy); 
			Uz += d_c[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(2*dz); 
		}
		//calculate scalar_operator 
		S = 0.5*(1+sqrtf(1-8*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		//update wavefield (internal part)
		Ux = 0.0f, Uy = 0.0f, Uz = 0.0f; //!!!first derivative Ux respresent second derivative Uxx 
		for(m=-M;m<=M;m++){
			Ux += d_c2[M+m]*u1[(idx+m)*ny*nz+idy*nz+idz]/(dx*dx);
			Uy += d_c2[M+m]*u1[idx*ny*nz+(idy+m)*nz+idz]/(dy*dy); 
			Uz += d_c2[M+m]*u1[idx*ny*nz+idy*nz+(idz+m)]/(dz*dz);
		}
		//update wave field 
		float taper=1.0/(1.0 + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]);//absorb layer coefficient

		u2[index] = (2*u1[index]-u2[index]*taper+ dt*dt*((1+2*d_epsilon[index])*(Ux+Uy)+Uz)*S*d_vp[index]*d_vp[index])*taper;
	}
}

#endif
//Forward
__global__ void TTI_forward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	/***** code struct 
		1.use shared memory load from u1(global) to  calculate Ux,Uy,Uz,Uxx,Uyy,Uzz
		2.use shared memory load from Ux(internal),ux(global) to  calculate Uxy,Uxz
		3.use shared memory load from Uy(internal),uy(global) to  calculate Uyz
		4.update wavefield 
	 ****/
	int idx =  blockDim.y * blockIdx.y + threadIdx.y;
	int idy =  blockDim.x * blockIdx.x + threadIdx.x;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S,epsilon,sinphi,sinthe,cosphi,costhe;
	float Uxx , Uyy , Uzz , Uxy, Uxz, Uyz;
	//int m;
	int index=idz*ny*nx+idx*ny+idy;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	__shared__ float uS[18][18][18];
	//1.1 Load u1 to shared memory uS
	if(index>=0&&index<(nx*ny*nz))
		uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=u1[index];
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =u1[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=u1[(idz+8)*ny*nx+idx*ny+idy];
		}
	}

	if(threadIdx.x<5){
		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =u1[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=u1[idz*ny*nx+idx*ny+idy+8];
		}
	}
	if(threadIdx.y<5){
		if(idz*ny*nx+(idx-5)*ny+idy>=0&&idz*ny*nx+(idx+8)*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y][threadIdx.x+5]    =u1[idz*ny*nx+(idx-5)*ny+idy];
			uS[threadIdx.z+5][threadIdx.y+8+5][threadIdx.x+5]=u1[idz*ny*nx+(idx+8)*ny+idy];
		}
	}
	__syncthreads();
	//1.2 Calculate the wavefield gradient  Ux,Uy,Uz,Uxx,Uyy,Uzz
	Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
	Uxx = 0.0f, Uyy = 0.0f, Uzz = 0.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
#pragma unroll
		for(index=-M;index<=M;index++){
			Ux+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(2*dx);
			Uy+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy) ;
			Uz+=d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz); 

			Uxx+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(dx*dx);
			Uyy+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+index+5]/(dy*dy); 
			Uzz+=d_c2[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(dz*dz);

		}
		index=idz*ny*nx+idx*ny+idy;
		ux[index] = Ux;
		uy[index] = Uy;	
	}
	__syncthreads();
	//2.1 Load Ux,ux to shared memory Us
	Uxy=0.0f, Uxz=0.0f, Uyz=0.0f;
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Ux;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =ux[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=ux[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	if(threadIdx.x<5){

		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =ux[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=ux[idz*ny*nx+idx*ny+idy+8];
		}
	}
	__syncthreads();
	//2.2 Calculate Uxy,Uxz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uxy += d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy);
		Uxz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}
	//3.1 Load Uy to shared memory Us
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Uy;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]=uy[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=uy[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	__syncthreads();
	//3.2 Calculate Uyz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uyz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}

	//4. Update wavefield
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		index=idz*ny*nx+idx*ny+idy;
		sinphi=sinf(d_phi[index]*3.1415926f/180);
		sinthe=sinf(d_the[index]*3.1415926f/180);
		cosphi=cosf(d_phi[index]*3.1415926f/180);
		costhe=cosf(d_the[index]*3.1415926f/180);

		epsilon =Ux,S=Uy;Ux=Uz;
		Uy= -sinphi * epsilon + cosphi * S;																//ky1
		Uz = cosphi * sinthe * epsilon + sinphi * sinthe * S+ costhe * Uz;//kz1
		Ux= cosphi * costhe *epsilon  +sinphi * costhe * S - sinthe * Ux; //kx1

		epsilon=d_epsilon[index];
		S = 0.5*(1+sqrtf(1-8*(epsilon-d_delta[index])*(Ux*Ux + Uy*Uy)*Uz*Uz/(((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));

		Uxx*=(1+2*epsilon*costhe*costhe*cosphi*cosphi+2*epsilon*sinphi*sinphi);
		Uyy*=(1+2*epsilon*costhe*costhe*sinphi*sinphi+2*epsilon*cosphi*cosphi);
		Uzz*=(1+2*epsilon*sinthe*sinthe);
		Uxz*=(-2*2*epsilon*sinthe*costhe*cosphi);
		Uxy*=(-2*2*epsilon*sinthe*sinthe*cosphi*sinphi);
		Uyz*=(-2*2*epsilon*sinthe*costhe*sinphi);

		if(idx == sx && idy == sy && idz == sz)
			u2[index] = 2*u1[index]-u2[index]+ dt*dt*(Uxx+Uyy+Uzz+Uxz+Uxy+Uyz)*S*d_vp[index]*d_vp[index] + wavelet;
		else
			u2[index] = (2*u1[index]-u2[index]*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz])) + dt*dt*(Uxx+Uyy+Uzz+Uxz+Uxy+Uyz)*S*d_vp[index]*d_vp[index])*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]));
	}
}
//Backward
__global__ void TTI_backward_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	/***** code struct 
		1.use shared memory load from u1(global) to  calculate Ux,Uy,Uz,Uxx,Uyy,Uzz
		2.use shared memory load from Ux(internal),ux(global) to  calculate Uxy,Uxz
		3.use shared memory load from Uy(internal),uy(global) to  calculate Uyz
		4.update wavefield 
	 ****/
	int idx =  blockDim.y * blockIdx.y + threadIdx.y;
	int idy =  blockDim.x * blockIdx.x + threadIdx.x;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S,epsilon,sinphi,sinthe,cosphi,costhe;
	float Uxx , Uyy , Uzz , Uxy, Uxz, Uyz;
	//int m;
	int index=idz*ny*nx+idx*ny+idy;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	__shared__ float uS[18][18][18];
	//1.1 Load u1 to shared memory uS
	if(index>=0&&index<(nx*ny*nz))
		uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=u1[index];
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =u1[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=u1[(idz+8)*ny*nx+idx*ny+idy];
		}
	}

	if(threadIdx.x<5){
		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =u1[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=u1[idz*ny*nx+idx*ny+idy+8];
		}
	}
	if(threadIdx.y<5){
		if(idz*ny*nx+(idx-5)*ny+idy>=0&&idz*ny*nx+(idx+8)*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y][threadIdx.x+5]    =u1[idz*ny*nx+(idx-5)*ny+idy];
			uS[threadIdx.z+5][threadIdx.y+8+5][threadIdx.x+5]=u1[idz*ny*nx+(idx+8)*ny+idy];
		}
	}
	__syncthreads();
	//1.2 Calculate the wavefield gradient  Ux,Uy,Uz,Uxx,Uyy,Uzz
	Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
	Uxx = 0.0f, Uyy = 0.0f, Uzz = 0.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
#pragma unroll
		for(index=-M;index<=M;index++){
			Ux+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(2*dx);
			Uy+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy) ;
			Uz+=d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz); 

			Uxx+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(dx*dx);
			Uyy+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+index+5]/(dy*dy); 
			Uzz+=d_c2[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(dz*dz);

		}
		index=idz*ny*nx+idx*ny+idy;
		ux[index] = Ux;
		uy[index] = Uy;	
	}
	__syncthreads();
	//2.1 Load Ux,ux to shared memory Us
	Uxy=0.0f, Uxz=0.0f, Uyz=0.0f;
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Ux;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =ux[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=ux[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	if(threadIdx.x<5){

		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =ux[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=ux[idz*ny*nx+idx*ny+idy+8];
		}
	}
	__syncthreads();
	//2.2 Calculate Uxy,Uxz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uxy += d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy);
		Uxz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}
	//3.1 Load Uy to shared memory Us
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Uy;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]=uy[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=uy[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	__syncthreads();
	//3.2 Calculate Uyz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uyz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}

	//4. Update wavefield
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		index=idz*ny*nx+idx*ny+idy;
		sinphi=sinf(d_phi[index]*3.1415926f/180);
		sinthe=sinf(d_the[index]*3.1415926f/180);
		cosphi=cosf(d_phi[index]*3.1415926f/180);
		costhe=cosf(d_the[index]*3.1415926f/180);

		epsilon =Ux,S=Uy;Ux=Uz;
		Uy= -sinphi * epsilon + cosphi * S;																//ky1
		Uz = cosphi * sinthe * epsilon + sinphi * sinthe * S+ costhe * Uz;//kz1
		Ux= cosphi * costhe *epsilon  +sinphi * costhe * S - sinthe * Ux; //kx1

		epsilon=d_epsilon[index];
		S = 0.5*(1+sqrtf(1-8*(epsilon-d_delta[index])*(Ux*Ux + Uy*Uy)*Uz*Uz/(((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));

		Uxx*=(1+2*epsilon*costhe*costhe*cosphi*cosphi+2*epsilon*sinphi*sinphi);
		Uyy*=(1+2*epsilon*costhe*costhe*sinphi*sinphi+2*epsilon*cosphi*cosphi);
		Uzz*=(1+2*epsilon*sinthe*sinthe);
		Uxz*=(-2*2*epsilon*sinthe*costhe*cosphi);
		Uxy*=(-2*2*epsilon*sinthe*sinthe*cosphi*sinphi);
		Uyz*=(-2*2*epsilon*sinthe*costhe*sinphi);

		u2[index] = (2*u1[index]-u2[index]*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz])) + dt*dt*(Uxx+Uyy+Uzz+Uxz+Uxy+Uyz)*S*d_vp[index]*d_vp[index])*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]));
	}
}
//Reconstruction
__global__ void TTI_reconstruction_wavefield_update(
		float *u1,float *u2,float *d_c,float *d_c2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *d_vp,float *d_epsilon,float* d_delta,float* d_phi,float* d_the,
		int sx,int sy,int sz,
		float *ux, float *uy, float *uz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	/***** code struct 
		1.use shared memory load from u1(global) to  calculate Ux,Uy,Uz,Uxx,Uyy,Uzz
		2.use shared memory load from Ux(internal),ux(global) to  calculate Uxy,Uxz
		3.use shared memory load from Uy(internal),uy(global) to  calculate Uyz
		4.update wavefield 
	 ****/
	int idx =  blockDim.y * blockIdx.y + threadIdx.y;
	int idy =  blockDim.x * blockIdx.x + threadIdx.x;
	int idz =  blockDim.z * blockIdx.z + threadIdx.z;
	float Ux,Uy,Uz,S,epsilon,sinphi,sinthe,cosphi,costhe;
	float Uxx , Uyy , Uzz , Uxy, Uxz, Uyz;
	//int m;
	int index=idz*ny*nx+idx*ny+idy;
	//initialize total scalar_operator = 1.0f;(initial value)
	S= 1.0f;
	__shared__ float uS[18][18][18];
	//1.1 Load u1 to shared memory uS
	if(index>=0&&index<(nx*ny*nz))
		uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=u1[index];
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =u1[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=u1[(idz+8)*ny*nx+idx*ny+idy];
		}
	}

	if(threadIdx.x<5){
		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =u1[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=u1[idz*ny*nx+idx*ny+idy+8];
		}
	}
	if(threadIdx.y<5){
		if(idz*ny*nx+(idx-5)*ny+idy>=0&&idz*ny*nx+(idx+8)*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y][threadIdx.x+5]    =u1[idz*ny*nx+(idx-5)*ny+idy];
			uS[threadIdx.z+5][threadIdx.y+8+5][threadIdx.x+5]=u1[idz*ny*nx+(idx+8)*ny+idy];
		}
	}
	__syncthreads();
	//1.2 Calculate the wavefield gradient  Ux,Uy,Uz,Uxx,Uyy,Uzz
	Ux = 0.0f, Uy = 0.0f, Uz = 0.0f;
	Uxx = 0.0f, Uyy = 0.0f, Uzz = 0.0f;
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
#pragma unroll
		for(index=-M;index<=M;index++){
			Ux+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(2*dx);
			Uy+=d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy) ;
			Uz+=d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz); 

			Uxx+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5+index][threadIdx.x+5]/(dx*dx);
			Uyy+=d_c2[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+index+5]/(dy*dy); 
			Uzz+=d_c2[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(dz*dz);

		}
		index=idz*ny*nx+idx*ny+idy;
		ux[index] = Ux;
		uy[index] = Uy;	
	}
	__syncthreads();
	//2.1 Load Ux,ux to shared memory Us
	Uxy=0.0f, Uxz=0.0f, Uyz=0.0f;
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Ux;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]    =ux[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=ux[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	if(threadIdx.x<5){

		if(idz*ny*nx+idx*ny+idy-5>=0&&idz*ny*nx+idx*ny+idy+8<(nx*ny*nz)){
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x]    =ux[idz*ny*nx+idx*ny+idy-5];
			uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+8+5]=ux[idz*ny*nx+idx*ny+idy+8];
		}
	}
	__syncthreads();
	//2.2 Calculate Uxy,Uxz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uxy += d_c[M+index]*uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5+index]/(2*dy);
		Uxz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}
	//3.1 Load Uy to shared memory Us
	uS[threadIdx.z+5][threadIdx.y+5][threadIdx.x+5]=Uy;
	if(threadIdx.z<5){
		if((idz-5)*ny*nx+idx*ny+idy>=0&&(idz+8)*ny*nx+idx*ny+idy<(nx*ny*nz)){
			uS[threadIdx.z][threadIdx.y+5][threadIdx.x+5]=uy[(idz-5)*ny*nx+idx*ny+idy];
			uS[threadIdx.z+8+5][threadIdx.y+5][threadIdx.x+5]=uy[(idz+8)*ny*nx+idx*ny+idy];
		}
	}
	__syncthreads();
	//3.2 Calculate Uyz
#pragma unroll
	for(index=-M;index<=M;index++){
		Uyz += d_c[M+index]*uS[threadIdx.z+5+index][threadIdx.y+5][threadIdx.x+5]/(2*dz);
	}

	//4. Update wavefield
	if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M){
		index=idz*ny*nx+idx*ny+idy;
		sinphi=sinf(d_phi[index]*3.1415926f/180);
		sinthe=sinf(d_the[index]*3.1415926f/180);
		cosphi=cosf(d_phi[index]*3.1415926f/180);
		costhe=cosf(d_the[index]*3.1415926f/180);

		epsilon =Ux,S=Uy;Ux=Uz;
		Uy= -sinphi * epsilon + cosphi * S;																//ky1
		Uz = cosphi * sinthe * epsilon + sinphi * sinthe * S+ costhe * Uz;//kz1
		Ux= cosphi * costhe *epsilon  +sinphi * costhe * S - sinthe * Ux; //kx1

		epsilon=d_epsilon[index];
		S = 0.5*(1+sqrtf(1-8*(epsilon-d_delta[index])*(Ux*Ux + Uy*Uy)*Uz*Uz/(((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1+2*epsilon)*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));

		Uxx*=(1+2*epsilon*costhe*costhe*cosphi*cosphi+2*epsilon*sinphi*sinphi);
		Uyy*=(1+2*epsilon*costhe*costhe*sinphi*sinphi+2*epsilon*cosphi*cosphi);
		Uzz*=(1+2*epsilon*sinthe*sinthe);
		Uxz*=(-2*2*epsilon*sinthe*costhe*cosphi);
		Uxy*=(-2*2*epsilon*sinthe*sinthe*cosphi*sinphi);
		Uyz*=(-2*2*epsilon*sinthe*costhe*sinphi);

		u2[index] = (2*u1[index]-u2[index]*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz])) + dt*dt*(Uxx+Uyy+Uzz+Uxz+Uxy+Uyz)*S*d_vp[index]*d_vp[index])*(1.0f/(1.0f + d_pmlx[idx] + d_pmly[idy] + d_pmlz[idz]));
	}
}
__global__ void extractRecordVSP(int nx, int ny, int nz, int pml, float *u2,  int gz, float *record){

	int idx =  blockDim.x * blockIdx.x + threadIdx.x;
	int idy =  blockDim.y * blockIdx.y + threadIdx.y;

	if(idx>=pml && idx<nx-pml && idy>=pml && idy<ny-pml)
		record[(idx-pml)*(ny-2*pml)+idy-pml]=u2[idx*ny*nz+idy*nz+gz];
}


