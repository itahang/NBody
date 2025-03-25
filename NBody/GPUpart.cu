#include"GPUpart.cuh"

__global__ void kernel(Body* d_pixels, int Width, int Height) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	// Ensure you stay within bounds
	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;
		int x = d_pixels[pixelIndex].position.x;
		int y = d_pixels[pixelIndex].position.x;

		if (x >= 1.0) {
			d_pixels[pixelIndex].position.x += -2.0;
		}
		d_pixels[pixelIndex].position.x += 0.005;
	}
}

