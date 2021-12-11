#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define GROUP_SIZE 2

__device__ int mandel(float c_re, float c_im, int maxIterations) {
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

    int now_x = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int now_y = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;
	
	if (now_x >= resX or now_y >= resY) return;
	
	for (int j = now_y; j < now_y + GROUP_SIZE; ++j) {
		if (j >= resY) break;

		for (int i = now_x; i < now_x + GROUP_SIZE; ++i) {
			if (i >= resX) break;

			float x = lowerX + i * stepX;
			float y = lowerY + j * stepY;

			int idx = i + j * resX;
			res_d[idx] = mandel(x, y, maxIterations);
		}
	}
}

#define BLOCK_SIZE 16

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int blk_x = (resX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blk_y = (resY + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int siz = resX * resY * sizeof(int);
    int* res;
    int* res_d;
	size_t pitch;
	
	//  cudaMallocPitch((void **)&d_data, &pitch, sizeof(float)*width, height);
	cudaMallocPitch((void**) &res_d, &pitch, sizeof(int) * resX, resY);
    cudaHostAlloc(&res, siz, cudaHostAllocMapped);

    // GPU Setting and Memory and Launch
    dim3 dim_block(BLOCK_SIZE / GROUP_SIZE, BLOCK_SIZE / GROUP_SIZE);
    dim3 dim_grid(blk_x, blk_y);

    mandelKernel <<<dim_grid, dim_block>>> (lowerX, lowerY, stepX, stepY, maxIterations, resX, resY, res_d);
	cudaDeviceSynchronize();

    cudaMemcpy(res, res_d, siz, cudaMemcpyDeviceToHost);
    memcpy(img, res, siz);

    cudaFree(res_d);
    cudaFreeHost(res);
}
