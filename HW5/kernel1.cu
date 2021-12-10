#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                                int maxIterations, int resX, int resY, int* res_d) {

    int now_x = blockIdx.x * blockDim.x + threadIdx.x;
    int now_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (now_x >= resX or now_y >= resY) return;

    float x = lowerX + now_x * stepX;
    float y = lowerY + now_y * stepY;

    int idx = now_x + now_y * resY;
    res_d[idx] = mandel(x, y, maxIterations);
}

#define BLOCK_SIZE 16

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int blk_x = ceil((float) resX / BLOCK_SIZE);
    int blk_y = ceil((float) resY / BLOCK_SIZE);

    int siz = resX * resY * sizeof(int);
    int* res = (int*) malloc(siz);
    int* res_d;
    cudaMalloc((void**) &res_d, siz);

    // GPU Setting and Memory and Launch
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(blk_x, blk_y);
    mandelKernel <<<dim_grid, dim_block>>> (lowerX, lowerY, stepX, stepY, maxIterations, resX, resY, res_d);

    cudaMemcpy(res, res_d, siz, cudaMemcpyDeviceToHost);
    memcpy(img, res, siz);

    cudafree(res_d);
    free(res);
}