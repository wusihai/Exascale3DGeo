/*************************************************************************
	> File Name: comm.h
	> Author: wusihai
	> Mail: wusihai18@gmail.com 
	> Created Time: Sat Jul 13 10:21:46 2019
 ************************************************************************/

#ifndef COMM_H
#define COMM_H
#include <time.h>
#include "slavertm.h"
#include "mpi.h"
//MPI参数
int process_size,my_rank,col_rank,row_rank;
int dims[3];
int nbrs[6];
int x_rank,y_rank,z_rank;
int tag; int ret;
MPI_Status status;
MPI_Request reqs[12];

#define X0 0
#define X1 1
#define Y0 2
#define Y1 3
#define Z0 4
#define Z1 5

//RTM参数
/*外部参数*/
int mgridnumx;
int mgridnumy; int mgridnumz;
int PMLX,PMLY,PMLZ;
int Nsx,Nsy;
int order;
float dt,dx,dy,dz;
float Fm;
int nsnap;
int Nt;

/*内部参数*/
int wave_len;
int nsgroup,sx,sy,sz,rz;
int nsstart,nsstop;
int it;
/*log 文件*/ FILE* flog;
char gptlfile[100];
time_t timep;
struct tm *p;

//模型数据
#define FILELEN 256
char vppath[FILELEN];
char epspath[FILELEN];
char deltapath[FILELEN];
char thepath[FILELEN];
char phipath[FILELEN];
char snap_file[FILELEN];
float* vpmodel;
float* vp_ext;
float* epsmodel;
float* eps_ext;
float* deltamodel;
float* delta_ext;
float* themodel;
float* the_ext;
float* phimodel;
float* phi_ext;
float  Vpmax;
//设备
int devicenum;
int mode;

//子波
float* wavelet;

float *record  ;//Vz 分量
//吸收边界
float *pmlx ;
float *pmly ;
float *pmlz ;
float *c  ;
float *c2 ;

int nx;
int ny;
int nz;
float *image;
float *illum;

#define v3d_ext(i,j,k) (i)*ny*nz+(j)*nz+k 
#define v3d(i,j,k) (i)*mgridnumy*mgridnumz+(j)*mgridnumz+k 
#endif

