/*************************************************************************
    > File Name: assistant.h
    > Author: wusihai
    > Mail: wusihai18@gmail.com 
    > Created Time: 2020年08月18日 星期二 12时05分17秒
 ************************************************************************/

#ifndef ASSISTANT_H
#define ASSISTANT_H

void gs(char* name);

void ge(char* name);
void Wavelet(float fm, float amp, float dt,  int nt, float *wavelet);
int fac(int n);

void coeff1d(float* x);

void coeff2d(float* x,float delta);

#endif

