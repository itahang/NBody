#include"GPUpart.cuh"

__global__ void kernel(float2* d_pixels, int Width, int Height) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	// Ensure you stay within bounds
	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;
		int x = d_pixels[pixelIndex].x;
		int y = d_pixels[pixelIndex].y;

		if (x >= 1.0) {
			d_pixels[pixelIndex].x = -1.0;
		}
		d_pixels[pixelIndex].x += 0.005;
	}
}

