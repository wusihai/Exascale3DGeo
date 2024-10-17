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
                hipLaunchKernelGGL(VTI_forward_wavefield_update, dim3(gridnew), dim3(blocknew), 0, 0, u1,u2,d_c,d_c2, nx, ny, nz,dx,dy,dz,dt,wavelet[k],d_vp,d_epsilon, d_delta,sx, sy, sz,d_pmlx,d_pmly,d_pmlz );

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

