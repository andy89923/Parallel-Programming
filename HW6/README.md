# Programming Assignment VI: OpenCL Programming
###### tags: `110Fall` `Parallel Programming` `HW`

#### **陳琮方 0816153**
[HackMD](https://hackmd.io/@CTFang/SyK9GLdcF)


## <font color="#0073FF"> Q1. Implementation of Image Convolution with OpenCL</font>

每個 Thread 同時算四個位置的的 convolution 值，並運用 Reference 所說的 float4 資料結構，可以同時讀取四個的值而不是一個一個讀。另外，我也使用 __constant 來存取 filter ，加快讀取的時間。在 hostFE 的地方在 createBuffer 的時候，我使用```CL_MEM_USE_HOST_PTR``` 讓 GPU 用 Host 端的的記憶體，減少搬動次數。


#### References:
- [image convolution optimization ](https://www.evl.uic.edu/kreda/gpu/image-convolution/)
- [High performance convolution](http://www.cmsoft.com.br/opencl-tutorial/case-study-high-performance-convolution-using-opencl-__local-memory/)


## <font color="#0073FF"> Q2. Using CUDA </font>

#### Implementation

我採用與 OpenCL 相同的作法，只是在 cuda 上不能用 float4 的運算規則，所以我將 float4 的運算展開。

#### Cuda vs OpenCL

![](https://i.imgur.com/41hI1Nn.png)

- Filter size
    - 1: 7 x 7
    - 2: 3 x 3
    - 3: 5 x 5


#### Result

cuda 在 hostFE 中，需要先配置記憶體(cudaMalloc)，再將要計算的東西搬到 Device 上，才能接下去做計算，所以在相同資料與計算方法中，會發現 cuda 執行時間稍稍比 OpenCL 來的慢。

#### Compile

[Ref1](https://forums.developer.nvidia.com/t/calling-a-function-located-in-cu-file-in-a-standard-c-file-in-visual-studio-2008/31462/6)

```Makefile```:
```Makefile
cuda: link.o $(OBJS)
        $(NVCC) -o cuda kernel.o $(OBJS) -lcudart -lcuda -arch=compute_61 -Xcompiler '-fPIC' -O3 -lOpenCL -m64

link.o:
        $(NVCC) $(CUDA_FLAGS) -dc kernel.cu -o kernel.o
```

#### Code
```cpp=1
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


extern "C" {
        void hostFEcuda(int filterWidth, float *filter, int imageHeight, int imageWidth,
                 float *inputImage, float *outputImage);
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
```