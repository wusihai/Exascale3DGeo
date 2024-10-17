#include "compress.h"

// 声明使用 zfp 进行数据压缩和解压缩
// 具体实现细节可参考 zfp 官网：https://zfp.llnl.gov/

double rate = 1;
double tolerance = 1.0e-6;
char* gboundary;

size_t csize1, csize2, csize3, csize4, csize5, csize6;

// 数据压缩函数，使用 zfp 库进行 3D 数据压缩
void cucompress(float* data, int n1, int n2, int n3, size_t* zfpsize, void* buffer) {
    // 具体实现已隐藏，详细内容参见 zfp 官网
}

// 数据解压缩函数，使用 zfp 库进行 3D 数据解压缩
void cudecompress(float* data, int n1, int n2, int n3, void* buffer) {
    // 具体实现已隐藏，详细内容参见 zfp 官网
}

// 使用指定精度进行数据压缩
void cucompressRecord(float* data, int n1, int n2, int n3, size_t* zfpsize, void* buffer) {
    // 具体实现已隐藏，详细内容参见 zfp 官网
}

// 使用指定精度进行数据解压缩
void cudecompressRecord(float* data, int n1, int n2, int n3, void* buffer) {
    // 具体实现已隐藏，详细内容参见 zfp 官网
}

void compressBoundary3d(int nx, int ny, int nz, float* data0, size_t* zfpsize, char* buffer) {
    cucompress(data0, 5, ny, nz, &csize1, buffer);
    cucompress(data0 + 5 * ny * nz, 5, ny, nz, &csize2, buffer + csize1);
    cucompress(data0 + 10 * ny * nz, 5, nx, ny, &csize3, buffer + csize1 + csize2);
    cucompress(data0 + 10 * ny * nz + 5 * nx * ny, 5, nx, ny, &csize4, buffer + csize1 + csize2 + csize3);
    cucompress(data0 + 10 * ny * nz + 10 * nx * ny, 5, nx, nz, &csize5, buffer + csize1 + csize2 + csize3 + csize4);
    cucompress(data0 + 10 * ny * nz + 10 * nx * ny + 5 * nx * nz, 5, nx, nz, &csize6, buffer + csize1 + csize2 + csize3 + csize4 + csize5);
    
    *zfpsize = csize1 + csize2 + csize3 + csize4 + csize5 + csize6;
}

void decompressBoundary3d(int nx, int ny, int nz, float* data0, char* buffer) {
    cudecompress(data0, 5, ny, nz, buffer);
    cudecompress(data0 + 5 * ny * nz, 5, ny, nz, buffer + csize1);
    cudecompress(data0 + 10 * ny * nz, 5, nx, ny, buffer + csize1 + csize2);
    cudecompress(data0 + 10 * ny * nz + 5 * nx * ny, 5, nx, ny, buffer + csize1 + csize2 + csize3);
    cudecompress(data0 + 10 * ny * nz + 10 * nx * ny, 5, nx, nz, buffer + csize1 + csize2 + csize3 + csize4);
    cudecompress(data0 + 10 * ny * nz + 10 * nx * ny + 5 * nx * nz, 5, nx, nz, buffer + csize1 + csize2 + csize3 + csize4 + csize5);
}
