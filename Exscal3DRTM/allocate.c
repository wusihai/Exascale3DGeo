/************************************************************************
**
** ���ļ����������ڴ�ռ������  
**
*************************************************************************/

#include "allocate.h"
#include "stdlib.h"
#include "stdio.h"

/************************************************************************/
/* ����1ά��������,����(����) */
/************************************************************************/
int* allocate1int(int column)
{
	int* p;
	int i;
	p =(int*)malloc(column*sizeof(int));
	if (p == 0)
	{
		free1int(p);
	}
	else
	{
		for (i=0;i<column;i++)
		{
			p[i] = 0.0;
		}
	}
	return p;
}

/************************************************************************/
/* �ͷ�1ά���������ڴ�ռ�,����(����ָ��) */
/************************************************************************/
void free1int(int* p)
{
	free(p);
	p = 0;
}


#if 0
/************************************************************************/
/* ����2ά��������,����(����,����) */
/************************************************************************/
int** allocate2int(int row,int column)
{
	int** p;
	p = new int* [row];
	for(int i=0;i<row;i++)
	{
		p[i] = new int [column];
	}
	if (p == 0)
	{
		free2int(p,row);
	}
	else
	{
		for (int i=0;i<row;i++)
		{
			for (int j=0;j<column;j++)
			{
				p[i][j] =0.0;
			}
		}
	}
	return p;
}

/************************************************************************/
/* ����3ά��������,����(ҳ��,����,����) */
/************************************************************************/
int*** allocate3int(int page,int row,int column)
{
	int i,j,k;
	int*** p;
	p = new int** [page];
	for(i=0; i<page; i++)
	{
		p[i] = new int* [row];
		for(j=0; j<row; j++) 
		{
			p[i][j]=new int[column]; 
		}
	}
	if (p == 0)
	{
		free3int(p,page,row);
	}
	else
	{
		for (i=0;i<page;i++)
		{
			for (j=0;j<row;j++)
			{
				for (k=0;k<column;k++)
				{
					p[i][j][k] = 0.0;;
				}
			}
		}
	}
	return p;
}
#endif
/************************************************************************/
/* ����1ά����������,����(����) */
/************************************************************************/
float* allocate1float(int column)
{
	int i;
	float* p;
	p = (float*)malloc(sizeof(float)*column);
	if (p == 0)
	{
		printf("allocate error\n");
		free1float(p);
	}
	else
	{
		for (i=0;i<column;i++)
		{
			p[i] = 0.0;
		}
	}
	return p;
}
#if 0
/************************************************************************/
/* ����2ά����������,����(����,����) */
/************************************************************************/
float** allocate2float(int row,int column)
{
	float** p;
	p = new float* [row];
	for(int i=0;i<row;i++)
	{
		p[i] = new float [column];
	}
	if (p == 0)
	{
		free2float(p,row);
	}
	else
	{
		for (int i=0;i<row;i++)
		{
			for (int j=0;j<column;j++)
			{
			    p[i][j] =0.0;
			}
		}
	}
	return p;
}

/************************************************************************/
/* ����3ά����������,����(ҳ��,����,����) */
/************************************************************************/
float*** allocate3float(int page,int row,int column)
{
	int i,j,k;
	float*** p;
	p = new float** [page];
	for(i=0; i<page; i++)
	{
		p[i] = new float* [row];
		for(j=0; j<row; j++) 
		{
			p[i][j]=new float[column]; 
		}
	}
	if (p == 0)
	{
		free3float(p,page,row);
	}
	else
	{
		for (i=0;i<page;i++)
		{
			for (j=0;j<row;j++)
			{
				for (k=0;k<column;k++)
				{
					p[i][j][k] = 0.0;;
				}
			}
		}
	}
	return p;
}

/************************************************************************/
/* ����1ά˫����������,����(����) */
/************************************************************************/
double* allocate1double(int column)
{
	double* p;
	p = new double [column];
	if (p == 0)
	{
		free1double(p);
	}
	else
	{
		for (int i=0;i<column;i++)
		{
			p[i] = 0.0;
		}
	}
	return p;
}

/************************************************************************/
/* ����2ά˫����������,����(����,����) */
/************************************************************************/
double** allocate2double(int row,int column)
{
	double** p;
	p = new double* [row];
	for(int i=0;i<row;i++)
	{
		p[i] = new double [column];
	}
	if (p == 0)
	{
		free2double(p,row);
	}
	else
	{
		for (int i=0;i<row;i++)
		{
			for (int j=0;j<column;j++)
			{
				p[i][j] =0.0;
			}
		}
	}
	return p;
}

/************************************************************************/
/* ����3ά˫����������,����(ҳ��,����,����) */
/************************************************************************/
double*** allocate3double(int page,int row,int column)
{
	int i,j,k;
	double*** p;
	p = new double** [page];
	for(i=0; i<page; i++)
	{
		p[i] = new double* [row];
		for(j=0; j<row; j++) 
		{
			p[i][j]=new double[column]; 
		}
	}
	if (p == 0)
	{
		free3double(p,page,row);
	}
	else
	{
		for (i=0;i<page;i++)
		{
			for (j=0;j<row;j++)
			{
				for (k=0;k<column;k++)
				{
					p[i][j][k] = 0.0;;
				}
			}
		}
	}
	return p;
}
/************************************************************************/
/* �ͷ�2ά���������ڴ�ռ�,����(����ָ��,����) */
/************************************************************************/
void free2int(int** p,int row)
{
	int i;
	for (i=0;i<row;i++)
	{
		delete [] p[i];  
	}
	delete [] p;  
	p = 0;
}

/************************************************************************/
/* �ͷ�3ά���������ڴ�ռ�,����(����ָ��,ҳ��,����) */
/************************************************************************/
void free3int(int*** p,int page,int row)
{
	int i,j;
	for(i=0; i<page; i++) 
	{
		for(j=0; j<row; j++) 
		{   
			delete [] p[i][j];   
		}   
	}       
	for(i=0; i<page; i++)   
	{       
		delete [] p[i];   
	}   
	delete [] p;  
	p = 0;
} 
#endif
/************************************************************************/
/* �ͷ�1ά�����������ڴ�ռ�,����(����ָ��) */
/************************************************************************/
void free1float(float* p)
{
	free(p);
	p = 0;
}
#if 0
/************************************************************************/
/* �ͷ�2ά�����������ڴ�ռ�,����(����ָ��,����) */
/************************************************************************/
void free2float(float** p,int row)
{
	int i;
	for (i=0;i<row;i++)
	{
		delete [] p[i];  
	}
	delete [] p;  
	p = 0;
}

/************************************************************************/
/* �ͷ�3ά�����������ڴ�ռ�,����(����ָ��,ҳ��,����) */
/************************************************************************/
void free3float(float*** p,int page,int row)
{
	int i,j;
	for(i=0; i<page; i++) 
	{
		for(j=0; j<row; j++) 
		{   
			delete [] p[i][j];   
		}   
	}       
	for(i=0; i<page; i++)   
	{       
		delete [] p[i];   
	}   
	delete [] p;  
	p = 0;
} 

/************************************************************************/
/* �ͷ�1ά˫�����������ڴ�ռ�,����(����ָ��) */
/************************************************************************/
void free1double(double* p)
{
	delete [] p;
	p = 0;
}

/************************************************************************/
/* �ͷ�2ά˫�����������ڴ�ռ�,����(����ָ��,����) */
/************************************************************************/
void free2double(double** p,int row)
{
	int i;
	for (i=0;i<row;i++)
	{
		delete [] p[i];  
	}
	delete [] p;  
	p = 0;
}

/************************************************************************/
/* �ͷ�3ά˫�����������ڴ�ռ�,����(����ָ��,ҳ��,����) */
/************************************************************************/
void free3double(double*** p,int page,int row)
{
	int i,j;
	for(i=0; i<page; i++) 
	{
		for(j=0; j<row; j++) 
		{   
			delete [] p[i][j];   
		}   
	}       
	for(i=0; i<page; i++)   
	{       
		delete [] p[i];   
	}   
	delete [] p;  
	p = 0;
} 


void zero1int( int* p, int column )
{
	for (int i=0; i<column; i++)
	{
		p[i]=0;
	}
}

void zero2int( int** p, int row, int column )
{
	for (int i=0; i<row; i++)
	{
		for (int j=0; j<column; j++)
		{
			p[i][j]=0;
		}
	}
}

void zero3int( int*** p, int page, int row, int column )
{
	for (int i=0; i<page; i++)
	{
		for (int j=0; j<row; j++)
		{
			for (int k=0; k<column; k++)
			{	  	
				p[i][j][k]=0;
			}
		}
	}
}

void zero1float( float* p, int column )
{
	for (int i=0; i<column; i++)
	{
		p[i]=0.0f;
	}
}

void zero2float( float** p, int row, int column )
{
	for (int i=0; i<row; i++)
	{
		for (int j=0; j<column; j++)
		{
			p[i][j]=0.0f;
		}
	}
}

void zero3float( float*** p, int page, int row, int column )
{
	for (int i=0; i<page; i++)
	{
		for (int j=0; j<row; j++)
		{
			for (int k=0; k<column; k++)
			{	  	
				p[i][j][k]=0.0f;
			}
		}
	}
}
#endif
