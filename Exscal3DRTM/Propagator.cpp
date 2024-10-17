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
		char processor_name[256];
		int name_len;
		MPI_Get_processor_name(processor_name, &name_len);

		printf("Device is not available in Processor %s\n",processor_name); exit(0);
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
	hipMalloc(&d_vp, sizeof(float)*nx*ny*nz);
	hipMalloc(&d_epsilon, sizeof(float)*nx*ny*nz);
	hipMalloc(&d_delta,   sizeof(float)*nx*ny*nz);

	if(mode==1){
		hipMalloc(&d_the, sizeof(float)*nx*ny*nz);
		hipMalloc(&d_phi, sizeof(float)*nx*ny*nz);
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
	hipMalloc(&u1, sizeof(float)*nx*ny*nz);
	hipMalloc(&u2, sizeof(float)*nx*ny*nz);
	hipMalloc(&ux, sizeof(float)*nx*ny*nz);
	hipMalloc(&uy, sizeof(float)*nx*ny*nz);
	//wavefield array for reconstruction
	float   *fu1, *fu2,   *fux, *fuy, *fuz;
	hipMalloc(&fu1, sizeof(float)*nx*ny*nz);
	hipMalloc(&fu2, sizeof(float)*nx*ny*nz);
	hipMalloc(&fux, sizeof(float)*nx*ny*nz);
	hipMalloc(&fuy, sizeof(float)*nx*ny*nz);
	//imaging array
	hipMalloc(&d_image, sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml));
	hipMalloc(&d_illum, sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml));

	//intialized memory
	hipMemset(u1, 0, sizeof(float)*nx*ny*nz);
	hipMemset(u2, 0, sizeof(float)*nx*ny*nz);
	hipMemset(ux, 0, sizeof(float)*nx*ny*nz);
	hipMemset(uy, 0, sizeof(float)*nx*ny*nz);
	hipMemset(d_image, 0, sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml));
	hipMemset(d_illum, 0, sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml));

	// copy data to device 
	hipMemcpy(d_c, c, (2*M+1)*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_c2, c2, (2*M+1)*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_vp, vp, sizeof(float)*nx*ny*nz, hipMemcpyHostToDevice);
	if(the!=NULL && phi!=NULL){
		hipMemcpy(d_the, the, sizeof(float)*nx*ny*nz, hipMemcpyHostToDevice);	
		hipMemcpy(d_phi, phi, sizeof(float)*nx*ny*nz, hipMemcpyHostToDevice);	
	}
	hipMemcpy(d_epsilon, epsilon,  sizeof(float)*nx*ny*nz, hipMemcpyHostToDevice);		
	hipMemcpy(d_delta, delta,      sizeof(float)*nx*ny*nz, hipMemcpyHostToDevice);

	hipMemcpy(d_pmlx, pmlx,  nx*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_pmly, pmly,  ny*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_pmlz, pmlz,   nz*sizeof(float), hipMemcpyHostToDevice);
	//hipMemcpy(d_record, record, nt*(nx-2*pml)*(ny-2*pml)*sizeof(float),hipMemcpyHostToDevice);

	hipGetLastError();
	hipDeviceSynchronize();

	//allocate two part to store boundary Async
	long size=1L*((nz-2*pml)*(ny-2*pml)+(nx-2*pml)*(ny-2*pml)+(nx-2*pml)*(nz-2*pml))*10;
	float *d_data0,*d_data1;
	char* buffer0,*buffer1;
	hipMalloc((void**)&d_data0,sizeof(float)*size);
	hipMalloc((void**)&d_data1,sizeof(float)*size);
	hipMalloc((void**)&buffer0,sizeof(float)*size);
	hipMalloc((void**)&buffer1,sizeof(float)*size);
	float* h_data0=(float*)malloc(sizeof(float)*size);
	float* h_data1=(float*)malloc(sizeof(float)*size);

	size_t csize;
	compressBoundary3d(nx-2*pml,ny-2*pml,nz-2*pml,d_data0,&csize,buffer0);
	char* gboundary=(char*)malloc(1L*csize*nt);
	if(my_rank==0){
		double s1=10*4*((nx-2*pml)*(ny-2*pml)+(nz-2*pml)*(ny-2*pml)+(nx-2*pml)*(nz-2*pml))/1024.0/1024;
		double s2=csize/1024./1024;
		printf("Orignal Data Size=%lfMB,After Compression Size=%lfMB\n",s1*nt,s2*nt);
	}

	//create muti-cudastream to overlap computer & memorycp kernels in GPU
	int ishot;
	for(ishot=shots;ishot<shote;ishot++){
		int ix=ishot%Nsx;
		int iy=ishot/Nsx;
		sx=SX[ix]+pml;
		sy=SY[iy]+pml;
		sz=SZ+pml;


		//sprintf(filename,"record/record%d.dat",ishot);
		//ifstream inF;
		//inF.open(filename, std::ifstream::binary);
		//inF.read((char*)(h_record),sizeof(float)*nt*(nx-2*pml)*(ny-2*pml) );
		//inF.close();
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
				hipLaunchKernelGGL(VTI_forward_wavefield_update, dim3(gridnew), dim3(blocknew), 0, 0, u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );
			//EX:exchange
			float* tmp=u1;
			u1 = u2;
			u2 = tmp;
			hipLaunchKernelGGL(extractRecordVSP, dim3(grid2), dim3(block2), 0, 0, nx, ny, nz, pml, u2,gz,d_record);

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
#if 1
		hipMemset(u1, 0, sizeof(float)*nx*ny*nz);
		hipMemset(u2, 0, sizeof(float)*nx*ny*nz);
		hipMemset(ux, 0, sizeof(float)*nx*ny*nz);
		hipMemset(uy, 0, sizeof(float)*nx*ny*nz);

		//begin the forward-propagation 
		for(k=0;k<nt;k++)
		{
			if(my_rank==0&&k%200==0){
				printf("forward propagation ishot%d nt = %d\n",ishot, k);
				fflush(stdout);
			}
			//wavefield update
			if(d_phi==NULL||d_the==NULL)
				hipLaunchKernelGGL(VTI_forward_wavefield_update, dim3(gridnew), dim3(blocknew), 0, 0, u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );

			//C2:output pml
			d_ul =d_data0;
			d_ur =d_ul +(nz-2*pml)*(ny-2*pml)*5;
			d_ut =d_ur +(nz-2*pml)*(ny-2*pml)*5;
			d_ub =d_ut +(nx-2*pml)*(ny-2*pml)*5;
			d_uf =d_ub +(nx-2*pml)*(ny-2*pml)*5;
			d_uba=d_uf +(nx-2*pml)*(nz-2*pml)*5;
			hipLaunchKernelGGL(wavefield_output_pmlx, dim3(gridX), dim3(blockX), 0, 0,  u1, u2, fu2, fu1, d_ul, d_ur,d_ut, d_ub, d_uf, d_uba, nx, ny, nz, pml);	
			hipLaunchKernelGGL(wavefield_output_pmly, dim3(gridY), dim3(blockY), 0, 0,  u1, u2, fu2, fu1, d_ul, d_ur,d_ut, d_ub, d_uf, d_uba, nx, ny, nz, pml);	
			hipLaunchKernelGGL(wavefield_output_pmlz, dim3(gridZ), dim3(blockZ), 0, 0,  u1, u2, fu2, fu1, d_ul, d_ur,d_ut, d_ub, d_uf, d_uba, nx, ny, nz, pml);	

			compressBoundary3d(nx-2*pml,ny-2*pml,nz-2*pml,d_data0,&csize,buffer0);
			hipMemcpy(gboundary+csize*k, buffer0, csize, hipMemcpyDeviceToHost);

			//EX:exchange
			float* tmp=u1;
			u1 = u2;
			u2 = tmp;

			if(k%nsnap==0&&k>0)
			{
				sprintf(snapname,"snap/shot%dit%d.dat",ishot+1,k);
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

		hipMemcpy(fu1,u2,sizeof(float)*nx*ny*nz,hipMemcpyDeviceToDevice);
		hipMemcpy(fu2,u1,sizeof(float)*nx*ny*nz,hipMemcpyDeviceToDevice);
		//intialized memory
		hipMemset(u1, 0, sizeof(float)*nx*ny*nz);
		hipMemset(u2, 0, sizeof(float)*nx*ny*nz);
		hipMemset(ux, 0, sizeof(float)*nx*ny*nz);
		hipMemset(uy, 0, sizeof(float)*nx*ny*nz);

		hipDeviceSynchronize();
		//begin the back-propagation and reconstruction//
		for(k=nt-1;k>=0;k--){
			if(my_rank==0&&k%200==0){
				printf("backward propagation ishot%d nt = %d\n",ishot, k);
				fflush(stdout);
			}
			//IO-create
			if(mode==0)
				hipLaunchKernelGGL(VTI_backward_wavefield_update, dim3(gridnew), dim3(blocknew), 0, 0,  u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,0,d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );

			//K1 (Load record)
			hipMemcpy(d_record,h_record+1L*k*(nx-2*pml)*(ny-2*pml), (nx-2*pml)*(ny-2*pml)*sizeof(float), hipMemcpyHostToDevice);

			hipLaunchKernelGGL(loadData, dim3((grid2)), dim3(dim3(block2)), 0, 0,  nx, ny, nz, pml,gz, u2,d_record);
			//output snapshot 
			if(k%nsnap==0&&k>0)
			{
				sprintf(snapname,"snap/shot%dit%d.dat-backward", ishot+1,k);
				hipMemcpy(h_u2, u2, nx*ny*nz*sizeof(float), hipMemcpyDeviceToHost);
				for(i=pml;i<nx-pml;i++)
					for(j=pml;j<ny-pml;j++)
						for(l=pml;l<nz-pml;l++)
							snap[(i-pml)*(ny-2*pml)*(nz-2*pml)+(j-pml)*(nz-2*pml)+l-pml] = h_u2[i*ny*nz+j*nz+l]; 

				ofstream ouF;
				ouF.open(snapname, std::ofstream::binary);
				ouF.write(reinterpret_cast<const char*>(snap),sizeof(float)*(ny-2*pml)*(nx-2*pml)*(nz-2*pml));
				ouF.close();
			}
			//C2 input PML
			hipMemcpy(buffer0,gboundary+(csize*k), csize ,hipMemcpyHostToDevice);
			decompressBoundary3d(nx-2*pml,ny-2*pml,nz-2*pml,d_data0,buffer0);

			//hipMemcpy(d_data0,gboundary+(csize*k), csize ,hipMemcpyHostToDevice);
			d_ul =d_data0;
			d_ur =d_ul +(nz-2*pml)*(ny-2*pml)*5;
			d_ut =d_ur +(nz-2*pml)*(ny-2*pml)*5;
			d_ub =d_ut +(nx-2*pml)*(ny-2*pml)*5;
			d_uf =d_ub +(nx-2*pml)*(ny-2*pml)*5;
			d_uba=d_uf +(nx-2*pml)*(nz-2*pml)*5;
			hipLaunchKernelGGL(addSourcePML, dim3((grid)), dim3((block)), 0, 0,  fu1, d_ul, d_ur, d_ut, d_ub, d_uf, d_uba, nx, ny, nz, pml);

			//C1 wavefield_update
			if(d_phi==NULL||d_the==NULL)
				hipLaunchKernelGGL(VTI_reconstruction_wavefield_update, dim3((gridnew)), dim3((blocknew)), 0, 0,  fu1,fu2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,0,d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );
			//EX-back:exchange
			float* tmp=u1;
			u1 = u2;
			u2 = tmp;

			//EX:exchange
			float* ftmp=fu1;
			fu1 = fu2;
			fu2 = ftmp;

			//output snapshot 
			if(k%nsnap==0&&k>0){
				sprintf(snapname,"snap/shot%dit%d.dat-Reconstruction",ishot+1,k);
				hipMemcpy(h_u2, fu2, nx*ny*nz*sizeof(float), hipMemcpyDeviceToHost);
				for(i=pml;i<nx-pml;i++)
					for(j=pml;j<ny-pml;j++)
						for(l=pml;l<nz-pml;l++)
							snap[(i-pml)*(ny-2*pml)*(nz-2*pml)+(j-pml)*(nz-2*pml)+l-pml] = h_u2[i*ny*nz+j*nz+l]; 
				ofstream ouF;
				ouF.open(snapname, std::ofstream::binary);
				ouF.write(reinterpret_cast<const char*>(snap),sizeof(float)*(ny-2*pml)*(nx-2*pml)*(nz-2*pml));
				ouF.close();
			}

			// Imaging 
			dim3 gridI((nz+Block_Sizez*2-1)/(2*Block_Sizez), (ny+Block_Sizey-1)/Block_Sizey, (nx+Block_Sizex-1)/Block_Sizex);
			dim3 blockI(Block_Sizez*2, Block_Sizey, Block_Sizex);
			hipLaunchKernelGGL(Imaging, dim3(dim3(gridI)), dim3(dim3(blockI)), 0, 0,  fu2, u2, d_image, d_illum, nx, ny, nz, pml);
		}

		//copy image & illumn to host
		hipMemcpy(h_u2, d_image,sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml), hipMemcpyDeviceToHost);
		////MPI_Allreduce(MPI_IN_PLACE,h_u2,(nx-2*pml)*(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		//for(int i=0;i<nx-2*pml;i++){
		//	if(my_rank==0) printf("Allreduce image %d / %d\n",i,nx-2*pml);
		//	MPI_Allreduce(MPI_IN_PLACE,h_u2+i*(ny-2*pml)*(nz-2*pml),(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		//}
		//if(my_rank==0){
		if(my_rank==0){ 
			printf("image write\n");
			fflush(stdout);
		}
		sprintf(snapname,"image/image%d.dat",ishot+1);
		printf("%s\n",snapname);	
		fflush(stdout);
		FILE* fp=fopen(snapname,"wb");
		fwrite(h_u2,sizeof(float),(nx-2*pml)*(ny-2*pml)*(nz-2*pml),fp);
		fclose(fp);
		//}

		hipMemcpy(h_u2, d_illum, sizeof(float)*(nx-2*pml)*(ny-2*pml)*(nz-2*pml), hipMemcpyDeviceToHost);
		//MPI_Allreduce(MPI_IN_PLACE,h_u2,(nx-2*pml)*(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		//for(int i=0;i<nx-2*pml;i++){
		//	if(my_rank==0) printf("Allreduce illum %d / %d\n",i,nx-2*pml);
		//	MPI_Allreduce(MPI_IN_PLACE,h_u2+i*(ny-2*pml)*(nz-2*pml),(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		//}

		//if(my_rank==0){
		if(my_rank==0){ 
			printf("illum write\n");
			fflush(stdout);
		}
		sprintf(snapname,"image/illum%d.dat",ishot+1);
		printf("%s\n",snapname);	
		fflush(stdout);
		fp=fopen(snapname,"wb");
		fwrite(h_u2,sizeof(float),(nx-2*pml)*(ny-2*pml)*(nz-2*pml),fp);
		fclose(fp);
		//}
#endif
	}
//	//copy image & illumn to host
//	hipMemcpy(image, d_image, (nx-2*pml)*(ny-2*pml)*(nz-2*pml)*sizeof(float), hipMemcpyDeviceToHost);
//	hipMemcpy(illum, d_illum, (nx-2*pml)*(ny-2*pml)*(nz-2*pml)*sizeof(float), hipMemcpyDeviceToHost);
//

	//free device memory
	hipFree(d_vp);hipFree(d_epsilon);hipFree(d_delta);hipFree(d_c);hipFree(d_c2);
	hipFree(d_pmlx);hipFree(d_pmly);hipFree(d_pmlz);
	hipFree(u1);hipFree(u2);
	hipFree(ux);hipFree(uy);
	hipFree(fu1);hipFree(fu2);
	hipFree(fux);hipFree(fuy);
	hipFree(d_image);
	hipFree(d_illum);
	hipFree(d_record);

	free(h_u2);
	free(snap); 
	hipFree(buffer0);
	hipFree(buffer1);

	hipFree(d_data0);
	hipFree(d_data1);
	free(h_data0);
	free(h_data1);
	free(gboundary);

}
