#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#pragma nvcc optimize("Xptxas", "O3")

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
	float new_re, new_im, tmp1, tmp2;

	int i;
    for (i = 0; i < maxIterations; ++i) {
		tmp1 = z_re * z_re;
		tmp2 = z_im * z_im;
        if (tmp1 + tmp2 > 4.f) break;

        new_re = tmp1 - tmp2;
        new_im = 2.f * z_re * z_im;
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

    int idx = now_x + now_y * resX;
    res_d[idx] = mandel(x, y, maxIterations);
}

#define BLOCK_SIZE 8

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int blk_x = (resX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blk_y = (resY + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int siz = resX * resY * sizeof(int);
    int* res_d;
    cudaMalloc((void**) &res_d, siz);

    // GPU Setting and Memory and Launch
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(blk_x, blk_y);
    mandelKernel <<<dim_grid, dim_block>>> (lowerX, lowerY, stepX, stepY, maxIterations, resX, resY, res_d);

    cudaMemcpy(img, res_d, siz, cudaMemcpyDeviceToHost);
    cudaFree(res_d);
}
