#include "compress.h"

double rate=1;
char* gboundary;

size_t csize1,csize2,csize3,csize4,csize5,csize6;
void cucompress(float* data,int n1,int n2,int n3,size_t* zfpsize,void*buffer){
	zfp_type type;     /* array scalar type */
	zfp_field* field;  /* array meta data */
	zfp_stream* zfp;   /* compressed stream */

	bitstream* stream; /* bit stream to write to or read from */

	type = zfp_type_float;
	//field = zfp_field_1d(data, type, n1);
	field = zfp_field_3d(data, type,n1,n2,n3);
	zfp = zfp_stream_open(NULL);
	zfp_stream_set_execution(zfp,zfp_exec_cuda);
	//zfp_stream_set_rate(zfp, rate, type, 1, 0); 
	zfp_stream_set_rate(zfp, rate, type, 3, 0); 

	/* allocate buffer for compressed data */
	size_t bufsize = zfp_stream_maximum_size(zfp, field);

	/* associate bit stream with allocated buffer */
	stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);
	*zfpsize = zfp_compress(zfp, field);
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);

}

void cudecompress(float* data,int n1,int n2,int n3,void* buffer){
	zfp_type type;     /* array scalar type */
	zfp_field* field;  /* array meta data */
	zfp_stream* zfp;   /* compressed stream */

	bitstream* stream; /* bit stream to write to or read from */

	type = zfp_type_float;
	field = zfp_field_3d(data, type,n1,n2,n3);
	//field = zfp_field_1d(data, type,n1);
	zfp = zfp_stream_open(NULL);

	zfp_stream_set_execution(zfp,zfp_exec_cuda);
	zfp_stream_set_rate(zfp, rate, type, 3, 0); 
	//zfp_stream_set_rate(zfp, rate, type, 1, 0); 

	/* allocate buffer for compressed data */
	size_t bufsize = zfp_stream_maximum_size(zfp, field); 
	/* associate bit stream with allocated buffer */
	stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);

	zfp_decompress(zfp, field);
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);

}
void compressBoundary3d(int nx,int ny,int nz,float* data0,size_t* zfpsize,char* buffer){

	cucompress(data0                          ,5,ny,nz,&csize1,buffer);	
	cucompress(data0+ 5*ny*nz                 ,5,ny,nz,&csize2,buffer+csize1);	
	cucompress(data0+10*ny*nz                 ,5,nx,ny,&csize3,buffer+csize1+csize2);	
	cucompress(data0+10*ny*nz+ 5*nx*ny        ,5,nx,ny,&csize4,buffer+csize1+csize2+csize3);	
	cucompress(data0+10*ny*nz+10*nx*ny        ,5,nx,nz,&csize5,buffer+csize1+csize2+csize3+csize4);	
	cucompress(data0+10*ny*nz+10*nx*ny+5*nx*nz,5,nx,nz,&csize6,buffer+csize1+csize2+csize3+csize4+csize5);	

	*zfpsize=csize1+csize2+csize3+csize4+csize5+csize6;
}
void decompressBoundary3d(int nx,int ny,int nz,float* data0,char* buffer){

	cudecompress(data0                          ,5,ny,nz,buffer);	
	cudecompress(data0+ 5*ny*nz                 ,5,ny,nz,buffer+csize1);	
	cudecompress(data0+10*ny*nz                 ,5,nx,ny,buffer+csize1+csize2);	
	cudecompress(data0+10*ny*nz+ 5*nx*ny        ,5,nx,ny,buffer+csize1+csize2+csize3);	
	cudecompress(data0+10*ny*nz+10*nx*ny        ,5,nx,nz,buffer+csize1+csize2+csize3+csize4);	
	cudecompress(data0+10*ny*nz+10*nx*ny+5*nx*nz,5,nx,nz,buffer+csize1+csize2+csize3+csize4+csize5);	

}

double tolerance=1.0e-6;
void cucompressRecord(float* data,int n1,int n2,int n3,size_t* zfpsize,void*buffer){
	zfp_type type;     /* array scalar type */
	zfp_field* field;  /* array meta data */
	zfp_stream* zfp;   /* compressed stream */

	bitstream* stream; /* bit stream to write to or read from */

	type = zfp_type_float;
	//field = zfp_field_1d(data, type, n1);
	field = zfp_field_3d(data, type,n1,n2,n3);
	zfp = zfp_stream_open(NULL);
	zfp_stream_set_execution(zfp,zfp_exec_serial);
	zfp_stream_set_accuracy(zfp,tolerance);
	//zfp_stream_set_rate(zfp, rate, type, 3, 0); 

	/* allocate buffer for compressed data */
	size_t bufsize = zfp_stream_maximum_size(zfp, field);

	/* associate bit stream with allocated buffer */
	stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);
	*zfpsize = zfp_compress(zfp, field);
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);
}

void cudecompressRecord(float* data,int n1,int n2,int n3,void* buffer){
	zfp_type type;     /* array scalar type */
	zfp_field* field;  /* array meta data */
	zfp_stream* zfp;   /* compressed stream */

	bitstream* stream; /* bit stream to write to or read from */

	type = zfp_type_float;
	field = zfp_field_3d(data, type,n1,n2,n3);
	zfp = zfp_stream_open(NULL);
	zfp_stream_set_execution(zfp,zfp_exec_serial);
	zfp_stream_set_accuracy(zfp,tolerance);
	//zfp_stream_set_rate(zfp, rate, type, 3, 0); 
	/* allocate buffer for compressed data */
	size_t bufsize = zfp_stream_maximum_size(zfp, field); 
	/* associate bit stream with allocated buffer */
	stream = stream_open(buffer, bufsize);
	zfp_stream_set_bit_stream(zfp, stream);
	zfp_stream_rewind(zfp);

	zfp_decompress(zfp, field);
	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);

}





#define COMPRESS
#ifdef COMPRESS
void *writebndry_work(void* arg0){
#if 0
	ioparameter* arg=(ioparameter*)arg0;
	sprintf(arg->filename,"boundary/PML-shot%dit%d", arg->ishot,arg->k-2);

	memcpy(gboundary+(csize*arg->k),arg->h_data,csize);

	//FILE*fp=fopen(arg->filename,"wb");
	////void*buffer=compress(arg->h_data,arg->size,&csize[arg->k-2]);
	//fwrite(arg->h_data,csize,1,fp);
	//fclose(fp);

	//free(buffer);

	//printf("ratio=%f,size=%ld,%ld,%ld\n",1.0*(arg->size*4)/csize,arg->size*4,bufsize,csize);
#endif
}
void *readbndry_work(void* arg0){
#if 0
	ioparameter* arg=(ioparameter*)arg0;
	sprintf(arg->filename,"boundary/PML-shot%dit%d", arg->ishot,arg->k-2);


	memcpy(arg->h_data,gboundary+(csize*arg->k),csize);

	//FILE*fp=fopen(arg->filename,"rb");

	////void*buffer=malloc(bufsize);
	//fread(arg->h_data,csize,1,fp);
	////decompress(arg->h_data,arg->size,buffer);

	//fclose(fp);
	//free(buffer);
#endif
}
#else
void *writebndry_work(void* arg0){
	ioparameter* arg=(ioparameter*)arg0;
	sprintf(arg->filename,"boundary/PML-shot%dit%d", arg->ishot,arg->k-2);
	FILE*fp=fopen(arg->filename,"wb");
	fwrite(arg->h_data,sizeof(float),(arg->size),fp);
	fclose(fp);
}
void *readbndry_work(void* arg0){
	ioparameter* arg=(ioparameter*)arg0;
	sprintf(arg->filename,"boundary/PML-shot%dit%d", arg->ishot,arg->k-2);
	FILE*fp=fopen(arg->filename,"rb");
	fread(arg->h_data,sizeof(float),(arg->size),fp);
	fclose(fp);
}
#endif


