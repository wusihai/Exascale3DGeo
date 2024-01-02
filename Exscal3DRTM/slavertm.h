/*************************************************************************
    > File Name: slave.h
    > Author: wusihai
    > Mail: wusihai18@gmail.com 
    > Created Time: Wed 05 Feb 2020 11:56:06 PM CST
 ************************************************************************/

#ifndef SLAVERTM_H
#define SLAVERTM_H
#include <stdio.h>
#include <string.h>

#define SIMD
#define SUNWAY

//定义结构体
typedef struct{
  //float *Pin,*Pout1,*Pout2,*Pout3;
  float *PT,*PTxx,*PTyy,*PTzz;
  float *PVx,*PVy,*PVz;
  float *Pvel;
  float dt,rho,dh,Vpmax,Vpconst;
  int stX,endX;
  int stY,endY;
  int stZ,endZ;
  int SubNx,SubNy,SubNz;
  int DivNx,DivNy,DivNz;
  int x_rank,y_rank,z_rank;
  int PMLX,PMLY,PMLZ;
  int mgridnumx,mgridnumy,mgridnumz;
} rtm_slave_parameter;


//定义结构体
typedef struct{
  float *Pout,*Pin1,*Pin2,*Pin3;
  int stX,endX;
  int stY,endY;
  int stZ,endZ;
  int SubNx,SubNy,SubNz;
} rtm_slave_merge;

//定义结构体
typedef struct{
  float *Pout[3],*Pin[3];
  int stX,endX;
  int stY,endY;
  int stZ,endZ;
  int SubNx,SubNy,SubNz;
  int DivNx,DivNy,DivNz;
  int PMLX,PMLY,PMLZ;
  int option;
} rtm_slave_iopack;
typedef struct{
  float *Pin[6];
  float *Pout[6];
  int stX,endX;
  int stY,endY;
  int stZ,endZ;
  int SubNx,SubNy,SubNz;
  int DivNx,DivNy,DivNz;
  int option;
} rtm_slave_mpipack;

typedef struct{
  float *Pout1,*Pout2,*Pin1,*Pin2;
  int stX,endX;
  int stY,endY;
  int stZ,endZ;
  int SubNx,SubNy,SubNz;
} rtm_slave_image;



#endif
