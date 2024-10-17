#ifndef ALLOCATE_H
#define ALLOCATE_H
/************************************************************************
**
** ���ӳ������������ڴ�ռ������  
**
************************************************************************/
//����1ά��������
int* allocate1int(int column);

//�ͷ�1ά��������ռ�
void free1int(int* p);


#if 0
//����2ά��������
int** allocate2int(int row,int column);

//����3ά��������
int*** allocate3int(int page,int row,int column);
#endif
//����1ά����������
float* allocate1float(int column);
#if 0
//����2ά����������
float** allocate2float(int row,int column);

//����3ά����������
float*** allocate3float(int page,int row,int column);

//����1ά˫����������
double* allocate1double(int column);

//����2ά˫����������
double** allocate2double(int row,int column);

//����3ά˫����������
double*** allocate3double(int page,int row,int column);
//�ͷ�2ά��������ռ�
void free2int(int** p,int row);

//�ͷ�3ά��������ռ�
void free3int(int*** p,int page,int row);
#endif
//�ͷ�1ά����������ռ�
void free1float(float* p);
#if 0
//�ͷ�2ά����������ռ�
void free2float(float** p,int row);

//�ͷ�3ά����������ռ�
void free3float(float*** p,int page,int row);

//�ͷ�1ά˫����������ռ�
void free1double(double* p);

//�ͷ�2ά˫����������ռ�
void free2double(double** p,int row);

//�ͷ�3ά˫����������ռ�
void free3double(double*** p,int page,int row);


//---------Assign the zero value to an array.---------//
//---assignment of int type array
void zero1int(int* p, int column);
void zero2int(int** p, int row, int column);
void zero3int(int*** p, int page, int row, int column);
//---assignment of float point type array
void zero1float(float* p, int column);
void zero2float(float** p, int row, int column);
void zero3float(float*** p, int page, int row, int column);
#endif
#endif
