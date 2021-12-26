#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, 
            cl_device_id *device, cl_context *context, cl_program *program) {

    cl_int status;
    cl_command_queue myqueue;
    cl_program myprog;
    cl_kernel kernel;
    cl_mem inp_dat, oup_dat, fil_dat;

    kernel = clCreateKernel(*program, "myConvolution", &status);
    myqueue = clCreateCommandQueue(*context, *device, 0, &status);

    int filt_size = filterWidth * filterWidth * sizeof(float);
    int data_size = imageHeight *  imageWidth * sizeof(float);
    int half_fitr = filterWidth / 2;

    inp_dat = clCreateBufer(*context,    CL_MEM_READ_ONLY, data_size, NULL, &status);
    oup_dat = clCreateBufer(*context,   CL_MEM_WRITE_ONLY, data_size, NULL, &status);
    fil_dat = clCreateBufer(*context, CL_MEM_USE_HOST_PTR, filr_size, NULL, &status);


    // set Argument
    // cl_int clSetKernelArg(kernel, arg_indx, arg_size, arg_value)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), inp_dat);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), oup_dat);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), fil_dat);

    clSetKernelArg(kernel, 3, sizeof(int), imageHeight);
    clSetKernelArg(kernel, 4, sizeof(int), imageWidth);
    clSetKernelArg(kernel, 5, sizeof(int), half_fitr);
    

    // set local and global workgroup sizes
    int grid_size = imageHeight * imageWidth / 4;
    // size_t loc_ws[2] = { 1, 1 };
    size_t glb_ws[2] = { grid_size, 1 };

    clEnqueueNDRangeKernel(myqueue, kernel, 2, 0, NULL, glb_ws, 0, NULL, NULL);
    clFinish(myqueue);
    clEnqueueReadBuffer(myqueue, out_dat, CL_TRUE, 0, data_size, ptr, 0, NULL, NULL);
}