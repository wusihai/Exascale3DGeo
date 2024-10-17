#ifndef ALLOCATE_H
#define ALLOCATE_H
/************************************************************************
**
** 该子程序用于数组内存空间的申请  
**
************************************************************************/
//申请1维整形数组
int* allocate1int(int column);

//释放1维整形数组空间
void free1int(int* p);


#if 0
//申请2维整形数组
int** allocate2int(int row,int column);

//申请3维整形数组
int*** allocate3int(int page,int row,int column);
#endif
//申请1维浮点形数组
float* allocate1float(int column);
#if 0
//申请2维浮点形数组
float** allocate2float(int row,int column);

//申请3维浮点形数组
float*** allocate3float(int page,int row,int column);

//申请1维双精度形数组
double* allocate1double(int column);

//申请2维双精度形数组
double** allocate2double(int row,int column);

//申请3维双精度形数组
double*** allocate3double(int page,int row,int column);
//释放2维整形数组空间
void free2int(int** p,int row);

//释放3维整形数组空间
void free3int(int*** p,int page,int row);
#endif
//释放1维浮点形数组空间
void free1float(float* p);
#if 0
//释放2维浮点形数组空间
void free2float(float** p,int row);

//释放3维浮点形数组空间
void free3float(float*** p,int page,int row);

//释放1维双精度形数组空间
void free1double(double* p);

//释放2维双精度形数组空间
void free2double(double** p,int row);

//释放3维双精度形数组空间
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
