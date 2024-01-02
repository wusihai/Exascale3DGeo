/*************************************************************************
    > File Name: rtmlib.h
    > Author: wusihai
    > Mail: wusihai18@gmail.com 
    > Created Time: Sat Jul 13 11:07:58 2019
 ************************************************************************/

#ifndef RTMLIB_H
#define RTMLIB_H

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

#include "comm.h"
#include "allocate.h"
#include "cdifferencefactor.h"

#define PI 3.14159265354

static  float StaCoef[5]={1.211242676,-0.089721680,0.013842773, -0.001765660, 0.000118679};//10阶

//初始化计算参数
void Init_parmter();
//初始化速度模型
void Init_model();
//PML初始化
void Init_PML();
//初始化从核参数
void Init_kernel_par();
//MPI模型分发
void distribute_model();
//三维模型边界扩充
void generate_v_3d(float *vp,int orgnx, int orgny,int orgnz,  int PMLx, int PMLy,int PMLz,  float *nvp);
//子波函数
int generate_wavelet();
int  wavelength_ensure(int recordpointnum, int main_fre, float dt, float coefficient/*=2.5*/);
void wavelet_riker0(int wave_len, float mainfre, float dt, float *wave);
//申请波场数据
void allocate_wavefield();
//申请通信buffer
void allocate_buffer();
//计算pml衰减参数
float pmlx_cal(int g_x);
float pmly_cal(int g_y);
float pmlz_cal(int g_z);
//加载震源
void add_souce(int sx,int sy,int sz,int it);
//合并应力场
void mergeStress();
void mergeStress_extrap();
//更新速度场
void update_Vel(); 
//更新正应力
void update_T();
//更新速度场-反传
void update_Vel_back(); 
//更新正应力-反传
void update_T_back();
//更新速度场-外推
void update_Vel_extrap(); 
//更新正应力-外推
void update_T_extrap();
/*正传-反传-通信V-Y-Z方向*/
void MPI_LaunchV();
void MPI_JoinV();
void MPI_LaunchT();
void MPI_JoinT();
/*外推-通信V-Y-Z方向*/
void MPI_LaunchV_extrap();
void MPI_JoinV_extrap();
void MPI_LaunchT_extrap();
void MPI_JoinT_extrap();

//保存速度边界值
void boundary_packT(int it);
void boundary_packV(int it);
void boundary_writeT(int it);
void boundary_writeV(int it);

/*边界读取+ 解包*/
void boundary_unpackT(int it);
void boundary_unpackV(int it);
void boundary_readT(int it );
void boundary_readV(int it);

//保存记录
void save_record(int it);	
void write_record(int ix,int iy);
//加载记录
void load_record(int it);	
//互相关成像
void imaging();
//叠加当前炮
void stack_image();
//图像滤波处理
void post_imaging();

//输出正传快照
void outforwardsnap(int it,int ix,int iy);
//输出正传快照
void outbackward_extrap(int it,int ix,int iy);
//输出当前成像结果
void outimage(int it,int ix,int iy);

//释放内存
void free_wavefield();
void free_buffer();

//GPTL interface
void gs(char* name);
void ge(char* name);
#endif
