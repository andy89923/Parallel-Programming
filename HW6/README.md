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

![](https://i.imgur.com/q37xggw.png)

- Filter size
    - 1: 7 x 7
    - 2: 3 x 3
    - 3: 5 x 5


#### Result

cuda 在 hostFE 中，需要先配置記憶體(cudaMalloc)，再將要計算的東西搬到 Device 上，才能接下去做計算，所以在相同資料與計算方法中，會發現 cuda 執行時間稍稍比 OpenCL 來的慢。