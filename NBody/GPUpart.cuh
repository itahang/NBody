#ifndef KERNEL_PART
#define KERNEL_PART

#include<cuda_runtime.h>
#include<device_launch_parameters.h>

extern __global__ void kernel(float2* d_pixels, int Width, int Height);

#endif // !KERNEL_PART


