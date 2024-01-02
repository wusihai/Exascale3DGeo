
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include "comm.h"
#include "allocate.h"
#include "rtmlib.h"
#include "gptl.h"
#include "assistant.h"
#define M 5
void rtm3d(int mode,int deviceid,int my_rank, int nx, int ny, int nz, int nt, int ntsnap, float dx, 
		float dy, float dz, float dt, int pml, int snapflag, int sx, int sy, int sz, int gz, 
		float *vp, float *epsilon, float *delta, float *the, float *phi, float *wavelet, float *pmlx, float *pmly, float *pmlz, float *c, float *c2,
		const char *snap_file, float *image, float *illum,int ret,
		int shots,int shote,int Nsx,int* SX,int* SY,int SZ,int wave_len);

//Source

int main(int argc,char**argv){
	int i,j,k;
	//初始化mpi环境
	MPI_Init(&argc,&argv);
	ret=GPTLinitialize();
	gs("Total");
	gs("Prepation");
	//获取进程数及进程id
	MPI_Comm_size(MPI_COMM_WORLD,&process_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	double T1,T2;
	T1=MPI_Wtime();
	//模型初始化及扩展模型-边界吸收
	Init_parmter();
	vpmodel=allocate1float((mgridnumx*mgridnumy*mgridnumz));
	vp_ext=allocate1float(((mgridnumx+2*PMLX)*(mgridnumy+2*PMLY)*(mgridnumz+2*PMLZ)));
	epsmodel=allocate1float((mgridnumx*mgridnumy*mgridnumz));
	eps_ext=allocate1float(((mgridnumx+2*PMLX)*(mgridnumy+2*PMLY)*(mgridnumz+2*PMLZ)));
	deltamodel=allocate1float((mgridnumx*mgridnumy*mgridnumz));
	delta_ext=allocate1float(((mgridnumx+2*PMLX)*(mgridnumy+2*PMLY)*(mgridnumz+2*PMLZ)));
	if(mode==1){
		themodel=allocate1float((mgridnumx*mgridnumy*mgridnumz));
		the_ext=allocate1float(((mgridnumx+2*PMLX)*(mgridnumy+2*PMLY)*(mgridnumz+2*PMLZ)));
		phimodel=allocate1float((mgridnumx*mgridnumy*mgridnumz));
		phi_ext=allocate1float(((mgridnumx+2*PMLX)*(mgridnumy+2*PMLY)*(mgridnumz+2*PMLZ)));
	}
	Init_model();
	generate_v_3d(vpmodel,mgridnumx,mgridnumy,mgridnumz, PMLX,PMLY,PMLZ, vp_ext);	
	generate_v_3d(epsmodel,mgridnumx,mgridnumy,mgridnumz, PMLX,PMLY,PMLZ, eps_ext);	
	generate_v_3d(deltamodel,mgridnumx,mgridnumy,mgridnumz, PMLX,PMLY,PMLZ, delta_ext);	

	if(mode==1){
		generate_v_3d(themodel,mgridnumx,mgridnumy,mgridnumz, PMLX,PMLY,PMLZ, the_ext);	
		generate_v_3d(phimodel,mgridnumx,mgridnumy,mgridnumz, PMLX,PMLY,PMLZ, phi_ext);	
	}
	free(vpmodel);
	free(epsmodel);
	free(deltamodel);
	if(mode==1){
		free(themodel);
		free(phimodel);
	}

	if(my_rank == 0){
		printf("Max Vp velocity is %.3fm/s, Suggested time interval dt=%.6f\n",Vpmax,dt);
		printf("Original model size      :mgridnumx=%d,mgridnumy=%d,mgridnumz=%d\n",mgridnumx,mgridnumy,mgridnumz);
		printf("Suggested absorbed layer :PMLX=%d,PMLY=%d,PMLZ=%d\n",PMLX,PMLY,PMLZ);
		printf("Total calculation size   :NX=%d,NY=%d,NZ=%d\n",mgridnumx+2*PMLX,mgridnumy+2*PMLY,mgridnumz+2*PMLZ);

		fprintf(flog,"Max Vp velocity is %.3f, Suggested time interval dt=%.6f\n",Vpmax,dt);
		fprintf(flog,"Original model size      :mgridnumx=%d,mgridnumy=%d,mgridnumz=%d\n",mgridnumx,mgridnumy,mgridnumz);
		fprintf(flog,"Suggested absorbed layer :PMLX=%d,PMLY=%d,PMLZ=%d\n",PMLX,PMLY,PMLZ);
		fprintf(flog,"Total calculation size   :NX=%d,NY=%d,NZ=%d\n",mgridnumx+2*PMLX,mgridnumy+2*PMLY,mgridnumz+2*PMLZ);
		fflush(flog);
	}

	//震源子波
	float* wavelet=allocate1float(Nt);
	Wavelet(Fm,1,dt,Nt,wavelet);

	image=allocate1float(nx*ny*nz);
	illum=allocate1float(nx*ny*nz);
	calcoef();
	calpml();

	//初始化震源位置
	int* SX=allocate1int(Nsx);
	int* SY=allocate1int(Nsy);
	int  SZ=sz;
	rz=rz+PMLZ;

	int ix,iy;
	for(ix=0;ix<Nsx;ix++){
		SX[ix]=(ix+1)*((mgridnumx-1)*1.0f/(Nsx+1));
	}
	for(iy=0;iy<Nsy;iy++){
		SY[iy]=(iy+1)*((mgridnumy-1)*1.0f/(Nsy+1));
	}
	if(my_rank==0){
		printf("Total shot X,Y is %d,%d \n",Nsx,Nsy);
		fprintf(flog,"Total shot X,Y is %d,%d \n",Nsx,Nsy);
		fflush(flog);
	}

	int dshot=(Nsx*Nsy)/process_size;
	int shots=my_rank*dshot;
	int shote=(my_rank+1)*dshot;

	int deviceid=my_rank%devicenum;
	rtm3d(mode,deviceid,my_rank, nx, ny, nz,Nt ,  nsnap,  dx, dy, dz, dt,  PMLX ,1, sx,sy, sz,  rz, vp_ext,eps_ext,delta_ext,the_ext, phi_ext,wavelet,pmlx,pmly,pmlz,c, c2,snap_file,image, illum,ret,shots,shote,Nsx,SX,SY,SZ,wave_len);

	//int pml=PMLX;
	//MPI_Allreduce(MPI_IN_PLACE,image,(nx-2*pml)*(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	//MPI_Allreduce(MPI_IN_PLACE,illum,(nx-2*pml)*(ny-2*pml)*(nz-2*pml),MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	//if(my_rank==0){
	//	FILE* fp=fopen("image/image.dat","wb");
	//	fwrite(image,sizeof(float),(nx-2*pml)*(ny-2*pml)*(nz-2*pml),fp);
	//	fclose(fp);
	//	fp=fopen("image/illum.dat","wb");
	//	fwrite(illum,sizeof(float),(nx-2*pml)*(ny-2*pml)*(nz-2*pml),fp);
	//	fclose(fp);
	//}

	T2=MPI_Wtime();
	if(my_rank==0) printf("Total Time=%lf\n",(T2-T1));
	//结束并行环境
	MPI_Finalize(); return 0; }

