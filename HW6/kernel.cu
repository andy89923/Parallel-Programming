#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convKernel(
    float* inp_dat, float* oup_dat, float* fil_dat,
    int imageHeight, int imageWidth, int half_fitr) {

    int poi_x = (blockIdx.x * blockDim.x + threadIdx.x);
    int poi_y = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    if (poi_x >= imageHeight || poi_y >= imageWidth) return;

    float4 ans = make_float4(0.0, 0.0, 0.0, 0.0), tmp, fil;

    int i, j, filter_idx = 0;
    int now_x, now_y, poi;
    for (i = -half_fitr; i <= half_fitr; ++i) {

        now_x = poi_x + i;
        if (now_x < 0 || now_x >= imageHeight) continue;
        
        int wx = now_x * imageWidth;
        for (j = -half_fitr; j <= half_fitr; ++j, ++filter_idx) {
            
            now_y = poi_y + j;
            if (now_y < 0 || now_y >= imageWidth) continue;

            poi = wx + now_y;
            
            tmp = make_float4(inp_dat[poi], inp_dat[poi+1], inp_dat[poi+2], inp_dat[poi+3]);
			float ff = fil_dat[filter_idx];
            fil = make_float4(ff, ff, ff, ff);

            ans.x += tmp.x * fil.x;
			ans.y += tmp.y * fil.y;
			ans.z += tmp.z * fil.z;
			ans.w += tmp.w * fil.w;
		}
	}
	int idx = poi_x * imageWidth + poi_y;
    oup_dat[idx + 0] = ans.x;
	oup_dat[idx + 1] = ans.y;
	oup_dat[idx + 2] = ans.z;
	oup_dat[idx + 3] = ans.w;
}


#define BLOCK_SIZE 16

void hostFEcuda(int filterWidth, float *filter, int imageHeight, int imageWidth,
                 float *inputImage, float *outputImage) {

    int blk_x = (imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blk_y = (imageWidth  + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int filt_size = filterWidth * filterWidth * sizeof(float);
    int data_size = imageHeight *  imageWidth * sizeof(float);
    int half_fitr = filterWidth / 2;

    float *ans, *fit, *inp;

    cudaMalloc((void**) &ans, data_size);
    cudaMalloc((void**) &inp, data_size);
    cudaMalloc((void**) &fit, filt_size);

    cudaMemcpy(inp, inputImage, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(fit, filter,     filt_size, cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_SIZE, 4);
    dim3 dim_grid(blk_x, blk_y);
    convKernel <<<dim_grid, dim_block>>> (inp, ans, fit, imageHeight, imageWidth, half_fitr);

    cudaMemcpy(outputImage, ans, data_size, cudaMemcpyDeviceToHost);

    cudaFree(ans);
    cudaFree(fit);
    cudaFree(inp);
}
