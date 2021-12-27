default: conv

# Cuda
NVCC = nvcc
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

# OpenCL
CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g -D CL_TARGET_OPENCL_VERSION=220

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

conv_cuda: $(OBJS) kernel.o 
	$(NVCC) $(CUDA_LINK_FLAGS) -c kernel.cu -o kernel.o 
	$(NVCC) $(CUDA_LINK_FLAGS) -o $@ kernel.o $(OBJS) -lOpenCL 

zip:
	rm -f HW6_0816153.zip
	zip HW6_0816153.zip kernel.cl hostFE.c url.txt

clean:
	rm -f conv *.o output.bmp ref.bmp
