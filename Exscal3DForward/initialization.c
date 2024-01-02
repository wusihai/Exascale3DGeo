/*************************************************************************
	> File Name: initialization.c
	> Author: wusihai
	> Mail: wusihai18@gmail.com 
	> Created Time: 三  1/29 12:53:40 2020
 ************************************************************************/

#include "rtmlib.h"


//初始化计算参数
void Init_parmter(){
	char buff[FILELEN];
	FILE* fp=fopen("./par/parameter.dat","r");
	fgets(buff,FILELEN,fp);
	fscanf(fp,"mode=%d\n",&mode);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"vppath=%s\n",vppath);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"epspath=%s\n",epspath);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"deltapath=%s\n",deltapath);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"thepath=%s\n",thepath);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"phipath=%s\n",phipath);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"Nsx=%d\n",&Nsx);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"Nsy=%d\n",&Nsy);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"mgridnumx=%d\n",&mgridnumx);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"mgridnumy=%d\n",&mgridnumy);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"mgridnumz=%d\n",&mgridnumz);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"sx=%d\n",&sx);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"sy=%d\n",&sy);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"sz=%d\n",&sz);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"rz=%d\n",&rz);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"PMLX=%d\n",&PMLX);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"PMLY=%d\n",&PMLY);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"PMLZ=%d\n",&PMLZ);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"Nt=%d\n",&Nt);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"nsnap=%d\n",&nsnap);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"dt=%f\n",&dt);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"dx=%f\n",&dx);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"dy=%f\n",&dy);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"dz=%f\n",&dz);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"Fm=%f\n",&Fm);
	fgets(buff,FILELEN,fp);
	fscanf(fp,"devicenum=%d\n",&devicenum);

	fclose(fp);
	//run log & Timing文件初始化
	time(&timep);
	p=gmtime(&timep);
	MPI_Bcast(&(p->tm_sec ),1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&(p->tm_min ),1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&(p->tm_hour),1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&(p->tm_mday),1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&(p->tm_mon ),1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&(p->tm_year),1,MPI_INT,0,MPI_COMM_WORLD);
	if(my_rank==0){
		sprintf(gptlfile,"./log/group%d.rtmrunlog.%d-%d-%d-%d-%d-%d",nsgroup,p->tm_year+1900,p->tm_mon+1,p->tm_mday,p->tm_hour+8,p->tm_min,p->tm_sec);
		flog=fopen(gptlfile,"w");
		fprintf(flog,"***************************************************\n");
		fprintf(flog,"                  3D pure qP wave RTM              \n");
		fprintf(flog,"                     MPI + FD                      \n");
		fprintf(flog,"                                                   \n");
		fprintf(flog,"***************************************************\n");
		fflush(flog);
	}


	nx=mgridnumx+2*PMLX;
	ny=mgridnumy+2*PMLY;
	nz=mgridnumz+2*PMLZ;
}

//PML初始化
void Init_PML(){
	int i;
	for(i=0;i<=30;i+=2){
		if((mgridnumx+2*PMLX)%dims[0] == 0){
			break;
		}
		PMLX++;
	} 
	for(i=0;i<=30;i+=2){
		if((mgridnumy+2*PMLY)%dims[1] == 0){
			break;
		}
		PMLY++;
	}
	for(i=0;i<=30;i+=2){
		if((mgridnumz+2*PMLZ)%dims[2] == 0){
			break;
		}
		PMLZ++;
	}
}

//初始化速度模型
void Init_model(){
	FILE* fp=fopen(vppath,"rb");
	if(fp==NULL){
		printf("open model failed!\n");
		fprintf(flog,"open model failed!\n");
		getchar();
	}
	fread(vpmodel,sizeof(float),(mgridnumx*mgridnumy*mgridnumz),fp);	
	fclose(fp);
	//根据稳定性条件反算时间采样间隔
	Vpmax=0.0f;
	int i;
	for(i=0;i<(mgridnumx*mgridnumy*mgridnumz);i++){
		if(Vpmax<vpmodel[i])
			Vpmax=vpmodel[i];
	}
	if(Vpmax>20000){
		printf("Max Vp error!\n");
		fprintf(flog,"Max Vp error!\n");
		getchar();
	}
	int L=5;
	float t_right1 = 0.0, t_left1 = 0.0;
	for (i = 1; i <= L; i++)
	{
		t_right1 += fabs(StaCoef[i - 1]);
	}
	t_right1 = 1 / t_right1;
	float dh=sqrt(dx*dx+dy*dy+dz*dz)/3;
	t_left1 = Vpmax*sqrtf((1 / dh)*(1 / dh) + (1 / dh)*(1 / dh));
	dt = (0.8)*t_right1 / t_left1;

	//读取各向异性参数
	fp=fopen(epspath,"rb");
	fread(epsmodel,sizeof(float),(mgridnumx*mgridnumy*mgridnumz),fp);	
	fclose(fp);

	fp=fopen(deltapath,"rb");
	fread(deltamodel,sizeof(float),(mgridnumx*mgridnumy*mgridnumz),fp);	
	fclose(fp);

	//读取角度
	if(mode==1){
		fp=fopen(thepath,"rb");
		fread(themodel,sizeof(float),(mgridnumx*mgridnumy*mgridnumz),fp);	
		fclose(fp);
		fp=fopen(phipath,"rb");
		fread(phimodel,sizeof(float),(mgridnumx*mgridnumy*mgridnumz),fp);	
		fclose(fp);
	}

}
//三维模型边界扩充
void generate_v_3d(float *vp,int orgnx, int orgny,int orgnz,  int PMLx, int PMLy,int PMLz,  float *nvp)
{
	int i,j,k;
	//原始模型
	for (i = PMLx; i < PMLx + orgnx; i++){
		for (j = PMLy; j < PMLy + orgny; j++){
			for (k = PMLz; k < orgnz + PMLz; k++){
				nvp[v3d_ext(i,j,k)] = vp[v3d((i - PMLx),(j - PMLy),(k - PMLz))];
			}
		}
	}
	//上边界
	for (i = PMLx; i < PMLx + orgnx; i++){
		for (j = 0; j < PMLy; j++){
			for (k = PMLz; k < orgnz + PMLz; k++){
				nvp[v3d_ext(i,j,k)] = vp[v3d((i - PMLx),(0),(k - PMLz))];
			}
		}
	}
	//下边界
	for (i = PMLx; i < PMLx + orgnx; i++){
		for (j = PMLy + orgny; j < 2 * PMLy + orgny; j++){
			for (k = PMLz; k < orgnz + PMLz; k++){
				nvp[v3d_ext(i,j,k)] = vp[v3d((i - PMLx),(orgny-1),(k - PMLz))];
			}
		}
	}
	//左边界
	for ( i = 0; i < PMLx; i++){
		for ( j = 0; j < 2 * PMLy + orgny; j++){
			for (k = PMLz; k < orgnz + PMLz; k++){
				nvp[v3d_ext(i,j,k)] = nvp[v3d_ext(PMLx,j,k)];
			}
		}
	}
	//右边界
	for (i = PMLx + orgnx; i < 2 * PMLx + orgnx; i++){
		for (j = 0; j < 2 * PMLy + orgny; j++){
			for (k = PMLz; k < orgnz + PMLz; k++){
				nvp[v3d_ext(i,j,k)] = nvp[v3d_ext((orgnx + PMLx - 1),j,k)];
			}
		}
	}
	/*前侧*/
	for ( i = 0; i < orgnx + 2 * PMLx; i++){
		for ( j = 0; j < orgny + 2 * PMLy; j++){
			for ( k = 0; k < PMLz; k++){
				nvp[v3d_ext(i,j,k)] = nvp[v3d_ext(i,j,PMLz)];
			}
		}
	}
	/*后侧*/
	for ( i = 0; i < orgnx + 2 * PMLx; i++){
		for ( j = 0; j < orgny + 2 * PMLy; j++){
			for ( k = orgnz + PMLz; k < orgnz + 2 * PMLz; k++){
				nvp[v3d_ext(i,j,k)] = nvp[v3d_ext(i,j,(orgnz + PMLz - 1))];
			}
		}
	}
}
//MPI模型分发
void distribute_model(){
#if 0
	//MPI分发
	vp_sub = allocate1float(DivNx*DivNy*DivNz);
	int i,j,k,x,y,p;
	for(x=0;x<DivNx;x++){
		for(y=0;y<DivNy;y++){
			//Process0
			if(my_rank==0)
				memcpy(&vp_sub[n3d(x,y,0)],&vp_ext[v3d_ext(x,y,0)],sizeof(float)*DivNz);
			//other Process
			for(p=1;p<dims[2]*dims[1]*dims[0];p++){
				i=p/(dims[1]*dims[2]);
				j=(p-i*dims[1]*dims[2])/dims[2];
				k=p-i*dims[1]*dims[2]-j*dims[2];
				if(my_rank==0)
					MPI_Send(&vp_ext[v3d_ext((i*DivNx+x),(j*DivNy+y),(k*DivNz))],DivNz,MPI_FLOAT,p,x,MPI_COMM_WORLD);
				if(my_rank==p)
					MPI_Recv(&vp_sub[n3d(x,y,0)],DivNz,MPI_FLOAT,0,x,MPI_COMM_WORLD,&status);
			}
		}
	}
	//释放全局模型内存
	if(my_rank==0){
		free1float(vp_ext);
		free1float(vpmodel);
	}
#endif
}
//子波函数
int generate_wavelet(){
	wave_len = wavelength_ensure(Nt, Fm, dt, 3.5);//确定子波长度
	wavelet = allocate1float(wave_len);//申请子波数组空间
	wavelet_riker0(wave_len, Fm, dt, wavelet);//计算子波序列
	return wave_len;
}

//确定子波长度
int  wavelength_ensure(int recordpointnum, int main_fre, float dt, float coefficient/*=2.5*/)//recordpointnum=nt=记录点数
{
	//coefficient=2.5~3.5
	int wave_len = (coefficient/(dt*main_fre));//根据子波主频确定长度,NINT为小数四舍五入。
	if(wave_len>recordpointnum) wave_len=recordpointnum;
	wave_len = (wave_len%2 == 0) ? wave_len+1 : wave_len;//子波确定为奇数个点
	return wave_len;
}

//创建零相位雷克子波
void wavelet_riker0(int wave_len, float mainfre, float dt, float *wave)//mainfre主频，dt采样间隔，wave子波
{
	float a=0;
	int t;
	if(wave_len%2!=0)//如果波长数为奇数
	{
		int half=wave_len/2;
		for (t=-half; t<=half; t++)
		{
			a=(float)pow(PI*mainfre*dt*t,2);
			wave[t+half]=(1-2*a)*exp(-a);
		}
	}
	else//如果波长为偶数
	{
		int half=wave_len/2;
		for (t=-half+1; t<=half; t++)
		{
			a=(float)pow(PI*mainfre*dt*t,2);
			wave[t+half-1]=((1-2*a)*exp(-a));
		}
	}
}
