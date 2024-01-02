/*************************************************************************
    > File Name: compress.h
    > Author: wusihai
    > Mail: wusihai18@gmail.com 
    > Created Time: 2020年08月31日 星期一 18时33分52秒
 ************************************************************************/
#ifndef COMPRESS_H
#define COMPRESS_H

#pragma once
#include"zfp.h"
#include "global.h"

extern double rate;
extern char* gboundary;
extern size_t csize1,csize2,csize3,csize4,csize5,csize6;

void cucompress(float* data,int n1,int n2,int n3,size_t* zfpsize,void*buffer);

void cudecompress(float* data,int n1,int n2,int n3,void* buffer);
void compressBoundary3d(int nx,int ny,int nz,float* data0,size_t* zfpsize,char* buffer);
void decompressBoundary3d(int nx,int ny,int nz,float* data0,char* buffer);

void *writebndry_work(void* arg0);
void *readbndry_work(void* arg0);
void cucompressRecord(float* data,int n1,int n2,int n3,size_t* zfpsize,void*buffer);


void cudecompressRecord(float* data,int n1,int n2,int n3,void* buffer);

#endif
