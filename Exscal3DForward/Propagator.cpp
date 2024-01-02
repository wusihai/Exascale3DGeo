#include "hip/hip_runtime.h"
/************ The program is writed by Lun Ruan, 2018.10***********************/
/*******3D Modeling for pure qP wave equation from Xu,2015************/

#include <thread>
#include  "mpi.h"
#include "global.h"
#include "GPU_kernel.h"
#include "gptl.h"
#include "allocate.h"
#include "zfp.h"
#include "iostream"
#include "fstream"
#include "compress.h"

using namespace std;
//Used in Pthread for Async
typedef struct _ioparameter{
	char* filename;
	float* h_data;
	int ishot;
	int k;
	long size;
} ioparameter;

__constant__ float d_cc[11];
__constant__ float d_c2c[11];

//Forward-
__global__ void VTI_forward_wavefield_update_2d(
		float* d_cc,float*d_c2c,
		float *   u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float * d_epsilon,float* d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void VTI_restruction_wavefield_update_2d(
		float* d_ccnew,float*d_c2cnew,
		float *   u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float *  d_epsilon,float*  d_delta,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		);
__global__ void VTI_backward_wavefield_update_2d(
		float* d_ccnew,float*d_c2cnew,
		float *   u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float *  d_epsilon,float*  d_delta,
		float *d_pmlx,float *d_pmly,float *d_pmlz ); 
//RTM3d 
extern "C"
void rtm3d(int mode,int deviceid,int my_rank, int nx, int ny, int nz, int nt,  int nsnap, float dx, 
		float dy, float dz, float dt, int pml, int snapflag, int sx, int sy, int sz, int gz, 
		float *vp, float *epsilon, float *delta, float *the, float *phi, float *wavelet,float *pmlx, float *pmly, float *pmlz, float *c, float *c2,
		const char *snap_file, float *image, float *illum,int ret,
		int shots,int shote,int Nsx,int* SX,int* SY,int SZ,int wave_len){

	//local index 
	int i,j,l,k;
	//time-assuming
	clock_t starttime, endtime;// t1, t2;
	//set device 
	int device_num;
	hipGetDeviceCount(&device_num);
	if(device_num > 0)
		hipSetDevice(deviceid);
	else
	{
		//printf("Device is not available\n"); exit(0);
		printf("Device is not available\n"); 
		return ;
	}
	char snapname[100];//for snapshot
	char filename[100];//for boundary 
	//cuda thread grid,block size
	dim3 grid((nx+Block_Sizex-1)/Block_Sizex, (ny+Block_Sizey-1)/Block_Sizey, (nz+Block_Sizez-1)/Block_Sizez);
	dim3 block(Block_Sizex, Block_Sizey, Block_Sizez);

	dim3 gridnew( (nz+Block_Sizez-1)/Block_Sizez,(ny+Block_Sizey-1)/Block_Sizey,(nx+Block_Sizex-1)/Block_Sizex );
	dim3 blocknew( Block_Sizez, Block_Sizey,Block_Sizex);

	dim3 grid2((nx+Block_Sizex-1)/Block_Sizex, (ny+Block_Sizey-1)/Block_Sizey);
	dim3 block2(Block_Sizex, Block_Sizey);

	dim3 block2new(BZ,BY);
	dim3 grid2new((nz+block2new.x-1)/(block2new.x), (ny+block2new.y-1)/(block2new.y));


	dim3 gridX(1, (ny+Block_Sizey-1)/Block_Sizey, (nz+Block_Sizez-1)/Block_Sizez);
	dim3 blockX(1,Block_Sizey, Block_Sizez);

	dim3 gridY((nx+Block_Sizex-1)/Block_Sizex, 1,(nz+Block_Sizez-1)/Block_Sizez);
	dim3 blockY(Block_Sizex, 1, Block_Sizez);
	dim3 gridZ((nx+Block_Sizex-1)/Block_Sizex, (ny+Block_Sizey-1)/Block_Sizey, 1);
	dim3 blockZ(Block_Sizex, Block_Sizey,1);

	//allocate host memory
	float	*snap       = (float*)malloc(sizeof(float)*((nx-2*pml)*(ny-2*pml)*(nz-2*pml)));
	float *h_u2       = (float*)malloc(sizeof(float)*(nx*ny*nz));
	size_t sizee=1L*sizeof(float)*(nx-2*pml)*(ny-2*pml)*nt;
	float *h_record  = (float*)malloc(sizee);

	/******* allocate device memory *****/
	float	*d_vp, *d_epsilon,*d_delta,*d_c,*d_c2,*d_pmlx,*d_pmly,*d_pmlz,
				*d_the, *d_phi,
				*d_ul, *d_ur, *d_ut, *d_ub,*d_uf, *d_uba,
				*d_image,*d_illum, *d_record,
				*u1, *u2, *ux, *uy, *uz;
	//different coef 
	hipMalloc(&d_c, (2*M+1)*sizeof(float));
	hipMalloc(&d_c2, (2*M+1)*sizeof(float));
	hipMemcpyToSymbol(d_c2c,c2,sizeof(float)*11 ,0,hipMemcpyHostToDevice);
	hipMemcpyToSymbol(d_cc ,c ,sizeof(float)*11 ,0,hipMemcpyHostToDevice);

	//model 
	hipMalloc(&d_vp, nx*ny*nz*sizeof(float));
	hipMalloc(&d_epsilon, nx*ny*nz*sizeof(float));
	hipMalloc(&d_delta, nx*ny*nz*sizeof(float));

	if(mode==1){
		hipMalloc(&d_the, nx*ny*nz*sizeof(float));
		hipMalloc(&d_phi, nx*ny*nz*sizeof(float));
	}
	else{
		d_the=NULL;d_phi=NULL;
	}

	//pml coef
	hipMalloc(&d_pmlx,nx*sizeof(float));
	hipMalloc(&d_pmly,ny*sizeof(float));
	hipMalloc(&d_pmlz, nz*sizeof(float));
	//the input data (receiver record)
	hipMalloc(&d_record, (nx-2*pml)*(ny-2*pml)*sizeof(float));
	//wavefield array for forward 
	hipMalloc(&u1, nx*ny*nz*sizeof(float));
	hipMalloc(&u2, nx*ny*nz*sizeof(float));
	hipMalloc(&ux, nx*ny*nz*sizeof(float));
	hipMalloc(&uy, nx*ny*nz*sizeof(float));
	//intialized memory
	hipMemset(u1, 0, nx*ny*nz*sizeof(float));
	hipMemset(u2, 0, nx*ny*nz*sizeof(float));
	hipMemset(ux, 0, nx*ny*nz*sizeof(float));
	hipMemset(uy, 0, nx*ny*nz*sizeof(float));

	// copy data to device 
	hipMemcpy(d_c, c, (2*M+1)*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_c2, c2, (2*M+1)*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_vp, vp, nx*ny*nz*sizeof(float), hipMemcpyHostToDevice);
	if(the!=NULL && phi!=NULL){
		hipMemcpy(d_the, the, nx*ny*nz*sizeof(float), hipMemcpyHostToDevice);	
		hipMemcpy(d_phi, phi, nx*ny*nz*sizeof(float), hipMemcpyHostToDevice);	
	}
	hipMemcpy(d_epsilon, epsilon,  nx*ny*nz*sizeof(float), hipMemcpyHostToDevice);		
	hipMemcpy(d_delta, delta, nx*ny*nz*sizeof(float), hipMemcpyHostToDevice);

	hipMemcpy(d_pmlx, pmlx,  nx*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_pmly, pmly,  ny*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_pmlz, pmlz,   nz*sizeof(float), hipMemcpyHostToDevice);
	//hipMemcpy(d_record, record, nt*(nx-2*pml)*(ny-2*pml)*sizeof(float),hipMemcpyHostToDevice);

	hipGetLastError();
	hipDeviceSynchronize();

	//create muti-cudastream to overlap computer & memorycp kernels in GPU
	int ishot;
	for(ishot=shots;ishot<shote;ishot++){
		int ix=ishot%Nsx;
		int iy=ishot/Nsx;
		sx=SX[ix]+pml;
		sy=SY[iy]+pml;
		sz=SZ+pml;

		hipMemset(u1, 0, nx*ny*nz*sizeof(float));
		hipMemset(u2, 0, nx*ny*nz*sizeof(float));
		hipMemset(ux, 0, nx*ny*nz*sizeof(float));
		hipMemset(uy, 0, nx*ny*nz*sizeof(float));
		for(k=0;k<nt;k++)
		{
			if(my_rank==0&&k%200==0){
				printf("ishot%d nt = %d\n",ishot, k);
				fflush(stdout);
			}

			//wavefield update
			if(d_phi==NULL||d_the==NULL)
				//hipLaunchKernelGGL(VTI_forward_wavefield_update, dim3(gridnew), dim3(blocknew), 0, 0, u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );
				hipLaunchKernelGGL(VTI_forward_wavefield_update_2d, dim3(dim3(grid2new)), dim3(dim3(block2new)), 0, 0, d_c,d_c2, u1,u2, nx, ny, nz,1.0f/dx,1.0f/dy,1.0f/dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );
			else
				hipLaunchKernelGGL(TTI_forward_wavefield_update, dim3(dim3(grid)), dim3(dim3(block)), 0, 0,  u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,d_phi,d_the,sx, sy, sz,ux,uy,uz,d_pmlx,d_pmly,d_pmlz );

			//EX:exchange
			float* tmp=u1;
			u1 = u2;
			u2 = tmp;
			hipLaunchKernelGGL(extractRecordVSP, dim3(dim3(grid2)), dim3(dim3(block2)), 0, 0,  nx, ny, nz, pml, u2,gz,d_record);

			hipMemcpy(h_record+1L*k*(nx-2*pml)*(ny-2*pml), d_record,(nx-2*pml)*(ny-2*pml)*sizeof(float), hipMemcpyDeviceToHost);

			if(k%nsnap==0&&k>0)
			{
				hipDeviceSynchronize();//synchronize GPU & IO 
				sprintf(snapname,"snap/shot%dit%d.dat_ff",ishot+1,k);
				hipMemcpy(h_u2, u2, nx*ny*nz*sizeof(float), hipMemcpyDeviceToHost);
				printf("%d,%d,%d\n",(ny-2*pml),(nx-2*pml),nz-2*pml);
				for(i=pml;i<nx-pml;i++)
					for(j=pml;j<ny-pml;j++)
						for(l=pml;l<nz-pml;l++)
							snap[(i-pml)*(ny-2*pml)*(nz-2*pml)+(j-pml)*(nz-2*pml)+l-pml] = h_u2[i*ny*nz+j*nz+l]; 
				ofstream ouF;
				ouF.open(snapname, std::ofstream::binary);
				ouF.write(reinterpret_cast<const char*>(snap), sizeof(float)*(ny-2*pml)*(nx-2*pml)*(nz-2*pml));
				ouF.close();
			}
		}
		sprintf(filename,"../ShotData/record%d.dat",ishot);
		FILE*ff=fopen(filename,"wb");
		fwrite(h_record,sizeof(float),1L*nt*(nx-2*pml)*(ny-2*pml),ff);
		fclose(ff);
	}

	//free device memory
	hipFree(d_vp);hipFree(d_epsilon);hipFree(d_delta);hipFree(d_c);hipFree(d_c2);
	hipFree(d_pmlx);hipFree(d_pmly);hipFree(d_pmlz);
	hipFree(u1);hipFree(u2);
	hipFree(ux);hipFree(uy);
	hipFree(d_record);

	free(h_u2);
	free(snap); 

	//free(gboundary);

}
//Forward-
__global__ void VTI_forward_wavefield_update_2d(
		float* d_ccnew,float*d_c2cnew,
		float *  u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float *  d_epsilon,float*  d_delta,
		int sx,int sy,int sz,
		float *d_pmlx,float *d_pmly,float *d_pmlz //能否直接计算，不从外部传入！
		){
	const	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	const	int idy =  blockDim.y * blockIdx.y + threadIdx.y;

	float xval[11];
	int index=0;
	__shared__ float uS[BY+2*M][BZ+2*M+1];//Ny-Nz

	for(int m=-M;m<=M;m++){
		xval[m+M]=u1[(M+m)*ny*nz+idy*nz+idz];
	}
	for(int idx=M;idx<nx-M;idx++){
		index=idx*ny*nz+idy*nz+idz;

		if(threadIdx.x<M)
			uS[threadIdx.y+M][threadIdx.x   ]=u1[idx*ny*nz+idy*nz+idz-M];
		if(threadIdx.x>=blockDim.x-M)
			uS[threadIdx.y+M][threadIdx.x+2*M]=u1[idx*ny*nz+idy*nz+idz+M];
		if(threadIdx.y<M)
			uS[threadIdx.y   ][threadIdx.x+M]=u1[idx*ny*nz+(idy-M)*nz+idz];
		if(threadIdx.y>=blockDim.y-M)
			uS[threadIdx.y+2*M][threadIdx.x+M]=u1[idx*ny*nz+(idy+M)*nz+idz];
		uS[threadIdx.y+M][threadIdx.x+M]=xval[M];

		__syncthreads();

		float eps=d_epsilon[index];
		float vp =d_vp[index];

		float Ux=d_ccnew[M]*xval[M];
		float Uy=d_ccnew[M]*xval[M];
		float Uz=d_ccnew[M]*xval[M];

		for(int m=1;m<=M;m++){
			Ux+=d_ccnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_ccnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]+uS[threadIdx.y+M-m][threadIdx.x+M]); 
			Uz+=d_ccnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}

		Ux=Ux*(0.5f*dx);
		Uy=Ux*(0.5f*dy);
		Uz=Ux*(0.5f*dz);

		//calculate scalar_operator 
		float S = 0.5f*(1.0f+sqrtf(1.0f-8.0f*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));

		Ux=d_c2cnew[M]*xval[M];
		Uy=d_c2cnew[M]*xval[M];
		Uz=d_c2cnew[M]*xval[M];

		for(int m=-M;m<0;m++){
			Ux+=d_c2cnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_c2cnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]  +uS[threadIdx.y+M-m][threadIdx.x+M])  ; 
			Uz+=d_c2cnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}
		Ux*=(dx*dx);
		Uy*=(dx*dx);
		Uz*=(dx*dx);

		float taper=1.0f/(1.0f+ d_pmlx[idx]+ d_pmly[idy]+ d_pmlz[idz]);//absorb layer coefficient
		if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M)
			if(idx == sx && idy == sy && idz == sz){
				u2[index] = 2.0f*xval[M]-u2[index]+ dt*dt*((1.0f+2.0f*eps)*(Ux+Uy)+Uz)*S*vp*vp + wavelet;
			}
			else{
				u2[index] = (2.0f*xval[M]-u2[index]*taper+ dt*dt*((1.0f+2.0f*eps)*(Ux+Uy)+Uz)*S*vp*vp)*taper;
			}
		for(int m=1;m<11;m++){
			xval[m-1]=xval[m];
		}
		xval[10]=u1[(idx+M+1)*ny*nz+idy*nz+idz];
	}
}
__global__ void VTI_restruction_wavefield_update_2d(
		float* d_ccnew,float*d_c2cnew,
		float *   u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float *  d_epsilon,float*  d_delta,
		float *d_pmlx,float *d_pmly,float *d_pmlz 
		){
	const	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	const	int idy =  blockDim.y * blockIdx.y + threadIdx.y;

	float xval[11];
	int index=0;
	__shared__ float uS[BY+2*M][BZ+2*M+1];//Ny-Nz

	for(int m=-M;m<=M;m++){
		xval[m+M]=u1[(M+m)*ny*nz+idy*nz+idz];
	}
	for(int idx=M;idx<nx-M;idx++){
		index=idx*ny*nz+idy*nz+idz;

		if(threadIdx.x<M)
			uS[threadIdx.y+M][threadIdx.x   ]=u1[idx*ny*nz+idy*nz+idz-M];
		if(threadIdx.x>=blockDim.x-M)
			uS[threadIdx.y+M][threadIdx.x+2*M]=u1[idx*ny*nz+idy*nz+idz+M];
		if(threadIdx.y<M)
			uS[threadIdx.y   ][threadIdx.x+M]=u1[idx*ny*nz+(idy-M)*nz+idz];
		if(threadIdx.y>=blockDim.y-M)
			uS[threadIdx.y+2*M][threadIdx.x+M]=u1[idx*ny*nz+(idy+M)*nz+idz];
		uS[threadIdx.y+M][threadIdx.x+M]=xval[M];

		__syncthreads();

		float eps=d_epsilon[index];
		float vp =d_vp[index];

		float Ux=d_ccnew[M]*xval[M];
		float Uy=d_ccnew[M]*xval[M];
		float Uz=d_ccnew[M]*xval[M];

		for(int m=1;m<=M;m++){
			Ux+=d_ccnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_ccnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]+uS[threadIdx.y+M-m][threadIdx.x+M]); 
			Uz+=d_ccnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}

		Ux=Ux*(0.5f*dx);
		Uy=Ux*(0.5f*dy);
		Uz=Ux*(0.5f*dz);

		//calculate scalar_operator 
		float S = 0.5f*(1.0f+sqrtf(1.0f-8.0f*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		Ux=d_c2cnew[M]*xval[M];
		Uy=d_c2cnew[M]*xval[M];
		Uz=d_c2cnew[M]*xval[M];

		for(int m=-M;m<0;m++){
			Ux+=d_c2cnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_c2cnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]  +uS[threadIdx.y+M-m][threadIdx.x+M])  ; 
			Uz+=d_c2cnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}
		Ux*=(dx*dx);
		Uy*=(dx*dx);
		Uz*=(dx*dx);

		float taper=1.0f/(1.0f+ d_pmlx[idx]+ d_pmly[idy]+ d_pmlz[idz]);//absorb layer coefficient
		if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M)
			u2[index] = (2.0f*xval[M]-u2[index]*taper+ dt*dt*((1.0f+2.0f*eps)*(Ux+Uy)+Uz)*S*vp*vp)*taper;

		for(int m=1;m<11;m++){
			xval[m-1]=xval[m];
		}
		xval[10]=u1[(idx+M+1)*ny*nz+idy*nz+idz];
	}
}
__global__ void VTI_backward_wavefield_update_2d(
		float* d_ccnew,float*d_c2cnew,
		float *   u1,float *u2,int nx,int ny,int nz,float dx,float dy,float dz,float dt,
		float wavelet,
		float *  d_vp,float *  d_epsilon,float*  d_delta,
		float *d_pmlx,float *d_pmly,float *d_pmlz 

		){

	const	int idz =  blockDim.x * blockIdx.x + threadIdx.x;
	const	int idy =  blockDim.y * blockIdx.y + threadIdx.y;

	float xval[11];
	int index=0;
	__shared__ float uS[BY+2*M][BZ+2*M+1];//Ny-Nz

	for(int m=-M;m<=M;m++){
		xval[m+M]=u1[(M+m)*ny*nz+idy*nz+idz];
		//xval[m+M]=tex1Dfetch(u1_tex,(M+m)*ny*nz+idy*nz+idz);
	}
	for(int idx=M;idx<nx-M;idx++){
		index=idx*ny*nz+idy*nz+idz;

		if(threadIdx.x<M)
			uS[threadIdx.y+M][threadIdx.x   ]=u1[idx*ny*nz+idy*nz+idz-M];
		if(threadIdx.x>=blockDim.x-M)
			uS[threadIdx.y+M][threadIdx.x+2*M]=u1[idx*ny*nz+idy*nz+idz+M];
		if(threadIdx.y<M)
			uS[threadIdx.y   ][threadIdx.x+M]=u1[idx*ny*nz+(idy-M)*nz+idz];
		if(threadIdx.y>=blockDim.y-M)
			uS[threadIdx.y+2*M][threadIdx.x+M]=u1[idx*ny*nz+(idy+M)*nz+idz];
		uS[threadIdx.y+M][threadIdx.x+M]=xval[M];

		__syncthreads();

		float eps=d_epsilon[index];
		float vp =d_vp[index];

		float Ux=d_ccnew[M]*xval[M];
		float Uy=d_ccnew[M]*xval[M];
		float Uz=d_ccnew[M]*xval[M];

		for(int m=1;m<=M;m++){
			Ux+=d_ccnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_ccnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]+uS[threadIdx.y+M-m][threadIdx.x+M]); 
			Uz+=d_ccnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}

		Ux=Ux*(0.5f*dx);
		Uy=Ux*(0.5f*dy);
		Uz=Ux*(0.5f*dz);

		//calculate scalar_operator 
		float S = 0.5f*(1.0f+sqrtf(1.0f-8.0f*(d_epsilon[index]-d_delta[index])*(Ux*Ux + Uy*Uy)*(Uz*Uz)/(((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS)*((1.0f+2.0f*d_epsilon[index])*(Ux * Ux + Uy * Uy) + Uz * Uz +EPS))));
		Ux=d_c2cnew[M]*xval[M];
		Uy=d_c2cnew[M]*xval[M];
		Uz=d_c2cnew[M]*xval[M];

		for(int m=-M;m<0;m++){
			Ux+=d_c2cnew[M+m]*(xval[M+m]+xval[M-m]);
			Uy+=d_c2cnew[M+m]*(uS[threadIdx.y+M+m][threadIdx.x+M]  +uS[threadIdx.y+M-m][threadIdx.x+M])  ; 
			Uz+=d_c2cnew[M+m]*(uS[threadIdx.y+M  ][threadIdx.x+M+m]+uS[threadIdx.y+M  ][threadIdx.x+M-m]); 
		}
		Ux*=(dx*dx);
		Uy*=(dx*dx);
		Uz*=(dx*dx);

		float taper=1.0f/(1.0f+ d_pmlx[idx]+ d_pmly[idy]+ d_pmlz[idz]);//absorb layer coefficient
		if(idx>=M && idx<nx-M && idy>=M && idy<ny-M && idz>=M && idz<nz-M)
			u2[index] = (2.0f*xval[M]-u2[index]*taper+ dt*dt*((1.0f+2.0f*eps)*(Ux+Uy)+Uz)*S*vp*vp)*taper;

		for(int m=1;m<11;m++){
			xval[m-1]=xval[m];
		}
		xval[10]=u1[(idx+M+1)*ny*nz+idy*nz+idz];
	}
}


