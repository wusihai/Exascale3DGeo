/*************************************************************************
  > File Name: rtmlib.c > Author: wusihai
  > Mail: wusihai18@gmail.com 
  > Created Time: Sat Jul 13 14:52:49 2019
 ************************************************************************/

#include "rtmlib.h"
#include "slavertm.h"

#if 0
//申请波场数据
void allocate_wavefield(){
  int size=SubNx*SubNy*SubNz;
  T   =allocate1float(size);
  /*正应力Txx三分量*/
  Txx   =allocate1float(size);
  /*正应力Tyy三分量*/
  Tyy   =allocate1float(size);
  /*正应力Tzz三分量*/
  Tzz   =allocate1float(size);
  /*速度Vx*/
  Vx   =allocate1float(size);
  /*速度Vy*/
  Vy   =allocate1float(size);
  /*速度Vz*/
  Vz   =allocate1float(size);
  /*外推波场*/

  extrapT   =allocate1float(size);
  /*正应力Txx三分量*/
  extrapTxx   =allocate1float(size);
  /*正应力Tyy三分量*/
  extrapTyy   =allocate1float(size);
  /*正应力Tzz三分量*/
  extrapTzz   =allocate1float(size);
  /*速度Vx*/
  extrapVx   =allocate1float(size);
  /*速度Vy*/
  extrapVy   =allocate1float(size);
  /*速度Vz*/
  extrapVz   =allocate1float(size);
  /*成像数据*/
  image=allocate1float(size);
  shot_energy=allocate1float(size);
  Total_image=allocate1float(size);
  /*地表记录*/
  size=DivNx*DivNy*Nt;
  recordVz   =allocate1float(size);
}
//申请通信buffer
void allocate_buffer(){
  int  size=5*DivNy*DivNz*6;
  sendX0    =allocate1float(size);
  recvX0    =allocate1float(size);

  size=5*DivNy*DivNz*3;
  iobufferX0V=allocate1float(size*Nf);
  iobufferX0T=allocate1float(size*Nf);

  size=DivNx*5*DivNz*6;
  sendY0    =allocate1float(size);
  recvY0    =allocate1float(size);

  size=DivNx*5*DivNz*3;
  iobufferY0V=allocate1float(size*Nf);
  iobufferY0T=allocate1float(size*Nf);

  size=DivNx*DivNy*5*6;
  sendZ0    =allocate1float(size);
  recvZ0    =allocate1float(size);

  size=DivNx*DivNy*5*3;
  iobufferZ0V=allocate1float(size*Nf);
  iobufferZ0T=allocate1float(size*Nf);
}
//计算PML衰减参数
float pmlx_cal(int g_x){
  float ddx = 0.0f;
  float R = 0.000001f;//理想边界反射系数
  if(g_x<PMLX){
	ddx= 3 * Vpmax / 2 / PMLX / dh*log(1 / R)*(PMLX - g_x) / PMLX*(PMLX - g_x) / PMLX;//上边吸收
  }
  else if(g_x >= (mgridnumx+2*PMLX) - PMLX){
	ddx = 3 * Vpmax / 2 / PMLX /dh*log(1 / R)*(g_x - PMLX - mgridnumx+1) / PMLX*(g_x - PMLX- mgridnumx+1) / PMLX;
  }
  return ddx;
}
float pmly_cal(int g_y){
  float ddy = 0.0f;
  float R = 0.000001f;//理想边界反射系数
  if(g_y<PMLY){
	ddy= 3 * Vpmax / 2 / PMLY / dh*log(1 / R)*(PMLY - g_y) / PMLY*(PMLY - g_y) / PMLY;//上边吸收
  }
  else if(g_y >= (mgridnumy+2*PMLY) - PMLY){
	ddy = 3 * Vpmax / 2 / PMLY /dh*log(1 / R)*(g_y - PMLY - mgridnumy+1) / PMLY*(g_y - PMLY- mgridnumy+1) / PMLY;
  }
  return ddy;
}
float pmlz_cal(int g_z){
  float ddz = 0.0f;
  float R = 0.000001f;//理想边界反射系数
  if(g_z<PMLZ){
	ddz= 3 * Vpmax / 2 / PMLZ / dh*log(1 / R)*(PMLZ - g_z) / PMLZ*(PMLZ - g_z) / PMLZ;//上边吸收
  }
  else if(g_z >= mgridnumz+PMLZ){
	ddz = 3 * Vpmax / 2 / PMLZ /dh*log(1 / R)*(g_z - PMLZ - mgridnumz+1) / PMLZ*(g_z - PMLZ- mgridnumz+1) / PMLZ;
  }
  return ddz;
}

//保存地表记录
void save_record(int it){
  int i,j;
  //Z方向
  if(z_rank==0){
	for(i=0;i<DivNx;i++){
	  for(j=0;j<DivNy;j++){
		//recordVz[i*DivNy*Nt+j*Nt+it]=Vz[n3d_ext((i+5),(j+5),(PMLZ+5+rz))];
		recordVz[i*DivNy*Nt+j*Nt+it]=Vz[n3d_ext((i+5),(j+5),(PMLZ+5+rz))]-recordVz[i*DivNy*Nt+j*Nt+it];
	  }
	}
  }
}
void write_record(int ix,int iy){
  //Z方向
  if(z_rank==0){
	char file_name[100];
	sprintf(file_name,"./snap/recordVz.%d.%d_%d_%d.dat",ix,iy,x_rank,y_rank);
	FILE*fp =fopen(file_name,"wb");
	fwrite(recordVz,sizeof(float),(DivNx*DivNy*Nt),fp);
	fclose(fp);
  }
}
//加载地表记录
void load_record(int it){
  int i,j;
  //Z方向
  if(z_rank==0){
	for(i=0;i<DivNx;i++){
	  for(j=0;j<DivNy;j++){
		extrapVz[n3d_ext((i+5),(j+5),(PMLZ+5+rz))]=recordVz[i*DivNy*Nt+j*Nt+it];
	  }
	}
  }
}

//互相关成像
void imaging(){
  int stX=5,endX=SubNx-5;
  int stY=5,endY=SubNy-5;
  int stZ=5,endZ=SubNz-5;
  int i,j,k;
	for ( k = stZ; k < endZ; k++){
		for ( i = stX; i < endX; i++){
			for ( j = stY; j < endY; j++){
				image[n3d_ext(i,j,k)]+=(extrapVz[n3d_ext(i,j,k)]*Vz[n3d_ext(i,j,k)]);
			}
		}
	}
	for ( k = stZ; k < endZ; k++){
		for ( i = stX; i < endX; i++){
			for ( j = stY; j < endY; j++){
				shot_energy[n3d_ext(i,j,k)]+=(Vz[n3d_ext(i,j,k)]*Vz[n3d_ext(i,j,k)]);
			}
		}
	}
}
//叠加当前炮
void stack_image(){
	int stX=5,endX=SubNx-5;
	int stY=5,endY=SubNy-5;
	int stZ=5,endZ=SubNz-5;
	int i,j,k;
	for ( k = stZ; k < endZ; k++){
		for ( i = stX; i < endX; i++){
			for ( j = stY; j < endY; j++){
				Total_image[n3d_ext(i,j,k)]+=image[n3d_ext(i,j,k)];
			}
		}
	}
}
//图像滤波处理
void post_imaging(int ix,int iy){
  //输出成像结果
  char file_name[100];
  sprintf(file_name,"./snap/Total_image.%d.%d_%d.dat",ix,iy,my_rank);
  FILE*fp =fopen(file_name,"wb");
  fwrite(Total_image,sizeof(float),(SubNx*SubNy*SubNz),fp);
  fclose(fp);
  sprintf(file_name,"./snap/energy.%d.%d_%d.dat",ix,iy,my_rank);
  fp =fopen(file_name,"wb");
  fwrite(shot_energy,sizeof(float),(SubNx*SubNy*SubNz),fp);
  fclose(fp);
}
//释放波场内存
void free_wavefield(){
  free1float(T);
  free1float(Txx);
  free1float(Tyy);
  free1float(Tzz);
  free1float(Vx);
  free1float(Vy);
  free1float(Vz);

  free1float(extrapT);
  free1float(extrapTxx);
  free1float(extrapTyy);
  free1float(extrapTzz);
  free1float(extrapVx);
  free1float(extrapVy);
  free1float(extrapVz);

  free1float(image);
  free1float(Total_image);
  free1float(shot_energy);
  free1float(recordVz);
}
//释放通信buffer
void free_buffer(){
  free1float(sendX0);
  free1float(recvX0);
  free1float(sendY0);
  free1float(recvY0);
  free1float(sendZ0);
  free1float(recvZ0);

  free1float(iobufferX0T);
  free1float(iobufferY0T);
  free1float(iobufferZ0T);
  free1float(iobufferX0V);
  free1float(iobufferY0V);
  free1float(iobufferZ0V);
}
//输出正传快照
void outforwardsnap(int it,int ix,int iy){
  if(it>0&&it%nsnap==0){
	char file_name[100];
	sprintf(file_name,"./snap/forward.%d.%d_%d_%d.dat",ix,iy,it,my_rank);
	FILE*fp =fopen(file_name,"wb");
	fwrite(Vz,sizeof(float),(SubNx*SubNy*SubNz),fp);
	fclose(fp);
  }
}
//输出正传快照
void outbackward_extrap(int it,int ix,int iy){
  if(it>0&&it%nsnap==0){
	char file_name[100];
	sprintf(file_name,"./snap/backward.%d.%d_%d_%d.dat",ix,iy,it,my_rank);
	FILE*fp =fopen(file_name,"wb");
	fwrite(Vz,sizeof(float),(SubNx*SubNy*SubNz),fp);
	fclose(fp);
  }
  if(it>0&&it%nsnap==0){
	char file_name[100];
	sprintf(file_name,"./snap/extrap.%d.%d_%d_%d.dat",ix,iy,it,my_rank);
	FILE*fp =fopen(file_name,"wb");
	fwrite(extrapVz,sizeof(float),(SubNx*SubNy*SubNz),fp);
	fclose(fp);
  }
}
//输出当前成像结果
void outimage(int it,int ix,int iy){
  if(it>0&&it%nsnap==0){
	char file_name[100];
	sprintf(file_name,"./snap/image.%d.%d_%d_%d.dat",ix,iy,it,my_rank);
	FILE*fp =fopen(file_name,"wb");
	fwrite(image,sizeof(float),(SubNx*SubNy*SubNz),fp);
	fclose(fp);
  } 
}
#endif
