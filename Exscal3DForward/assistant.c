/*************************************************************************
	> File Name: assistant.c
	> Author: wusihai
	> Mail: wusihai18@gmail.com 
	> Created Time: 三  1/29 12:51:57 2020
 ************************************************************************/

#include <stdio.h>
#include "gptl.h"
#include "allocate.h"
#include "comm.h"
#include "math.h"
#define M 5
#define PI 3.14159265354
void gs(char* name){
	GPTLstart(name);
}
void ge(char* name){
	GPTLstop(name);
}
void Wavelet(float fm, float amp, float dt,  int nt, float *wavelet)
{
	int i,j,k;
	float temp ,t;

	for(k=0;k<nt;k++)
	{
		t = k*dt ;
		temp = pow(PI*fm*(t-1/fm),2);
		wavelet[k] = (1-2*temp)*exp(-temp);

	}

}
int fac(int n)
	/*< fac for FD coefficent calculation >*/
{
	int s;

	if(n==0)
		return 1;
	s=1;
	while(n>0)
	{
		s=s*n;
		n--;
	}
	return s;
}
void coeff1d(float* x)
	/*< mix-th order coefficients for 1st-order spatial derivatives >*/
{
	int i;
	for(i=-M;i<=M;i++)
	{
		if(i==0)
			x[i+M]=0;
		else
			x[i+M]=2*fac(M)*fac(M)*pow(-1,i+1)/(i*fac(i+M)*fac(M-i));
	}

}
void coeff2d(float* x,float delta)    
	/*< m-th order coefficients for 2nd-order spatial derivatives, e.g. displacement-time equation >*/
{
	int mm=M+1;       
	int i,j,k=0,s=0;
	float max,t,q,sum=0;

	float A[mm][mm+1];

	for(i=0;i<mm;i++)   
	{
		A[i][mm-1]=0;
		A[i][mm]=0;
	}
	A[0][mm-1]=1;
	A[1][mm]=1;

	for(i=0;i<mm;i++)
		for(j=0;j<mm-1;j++)
		{
			A[i][j]=2*pow((float)(mm-1-j)*delta,2*i)/fac(2*i);
		}
	while(k<mm)
	{
		max=-9999;
		for(i=k;i<mm;i++)
			if(A[i][k]>max)
			{
				max=A[i][k];
				s=i;
			}                  
		for(j=k;j<=mm;j++)
		{
			t=A[k][j];
			A[k][j]=A[s][j];
			A[s][j]=t;
		}                            
		for(i=k+1;i<mm;i++)
		{
			q=A[i][k]/A[k][k];
			for(j=k;j<=mm;j++)
				A[i][j]=A[i][j]-q*A[k][j];
		}
		k++;
	}

	for(i=mm-1;i>=0;i--)
	{
		sum=0;
		for(j=i+1;j<mm;j++)
			sum+=A[i][j]*x[j];
		x[i]=(A[i][mm]-sum)/A[i][i];   
		x[2*mm-2-i]=x[i];
	}

	//      free(*A);
}
void calpml(){
	int i,j,k;
	pmlx = (float*)malloc(nx * sizeof (float));
	pmly = (float*)malloc(ny * sizeof (float));
	pmlz = (float*)malloc(nz * sizeof (float));
	for(i=0;i<nx;i++)
		pmlx[i]=0.0;
	float tmp, Refl_coef;
	tmp=(float)(PMLX)*dx;
	float base=0.00001;
	Refl_coef = logf(1.0f/base)*Vpmax*1.5/(tmp*tmp*tmp)*dt*0.5;
	for(i=0;i<PMLX;i++){
		tmp=(PMLX-i)*dx;
		pmlx[i]=Refl_coef*tmp*tmp;
		pmlx[nx-i-1]=Refl_coef*tmp*tmp;
		//pmlx[i]=     2*Vpmax/(PMLX*dx)*pow(tmp/(PMLX*dx),3)*logf(1.0/base)*dt;
		//pmlx[nz-i-1]=2*Vpmax/(PMLX*dx)*pow(tmp/(PMLX*dx),3)*logf(1.0/base)*dt;
	}

	for(i=0;i<ny;i++)
		pmly[i]=0.0;
	tmp=(float)PMLY*dy;
	Refl_coef = logf(1.0f/base)*Vpmax*1.5/(tmp*tmp*tmp)*dt*0.5;
	for(i=0;i<PMLY;i++){
		tmp=(PMLY-i)*dy;
		pmly[i]=Refl_coef*tmp*tmp;
		pmly[ny-i-1]=Refl_coef*tmp*tmp;
		//pmly[i]=2*Vpmax/(PMLY*dy)*pow(tmp/(PMLY*dy),3)*logf(1.0/base)*dt;
		//pmly[nz-i-1]=2*Vpmax/(PMLY*dy)*pow(tmp/(PMLY*dy),3)*logf(1.0/base)*dt;
	}

	for(i=0;i<nz;i++)
		pmlz[i]=0.0;
	tmp=(float)(PMLZ)*dz;
	Refl_coef = logf(1.0f/base)*Vpmax*1.5/(tmp*tmp*tmp)*dt*0.5;
	for(i=0;i<PMLZ;i++){
		tmp=(PMLZ-i)*dz;
		pmlz[i]=2*Vpmax/(PMLZ*dz)*pow(tmp/(PMLZ*dz),3)*logf(1.0/base)*dt;
		pmlz[nz-i-1]=2*Vpmax/(PMLZ*dz)*pow(tmp/(PMLZ*dz),3)*logf(1.0/base)*dt;
		//pmlz[i]=Refl_coef*tmp*tmp;
		//pmlz[nz-i-1]=Refl_coef*tmp*tmp;
	}
}
void calcoef(){
	c = allocate1float(2*M+1);
	c2= allocate1float(2*M+1);
	coeff1d(c);
	coeff2d(c2,1.0);
}
