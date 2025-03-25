#ifndef KERNEL_PART
#define KERNEL_PART

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"Body.cu"

extern __global__ void kernel(Body* d_pixels, int Width, int Height);

#endif // !KERNEL_PART


