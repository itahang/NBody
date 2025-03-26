#include"GPUpart.cuh"
#include<cuda_runtime.h>

__device__ float2 calcAcceleration(Body* s, Body* t, float epsilon, float rfactor = 1000) {
	float2 r = { rfactor * (t->position.x - s->position.x),rfactor * (t->position.y - s->position.y) };
	float dist_sq = r.x * r.x + r.y * r.y + epsilon;  // Add softening factor
	float inv_dist = rsqrtf(dist_sq);  // Fast inverse square root
	float inv_dist3 = inv_dist * inv_dist * inv_dist;  // 1/d³

	float force_mag = s->mass * t->mass * inv_dist3;
	return { force_mag * r.x, force_mag * r.y };
}


__global__ void kernel(Body* d_pixels, int Width, int Height) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	// considerting dt=0.001
	float dt = 0.001;
	float epsilon = 1e-4;
	// Ensure you stay within bounds
	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;
		d_pixels[pixelIndex].acceleration = { 0, 0 };
		for (int i = 0; i < Width * Height; i++) {

			if (i == pixelIndex) {
				continue;
			}

			float2 tAcc = calcAcceleration(&d_pixels[pixelIndex], &d_pixels[i], epsilon);
			d_pixels[pixelIndex].acceleration.x += tAcc.x;
			d_pixels[pixelIndex].acceleration.y += tAcc.y;
		}


		// Update velocity
		d_pixels[pixelIndex].velocity.x += dt * d_pixels[pixelIndex].acceleration.x;
		d_pixels[pixelIndex].velocity.y += dt * d_pixels[pixelIndex].acceleration.y;
		// Update position
		d_pixels[pixelIndex].position.x += dt * d_pixels[pixelIndex].velocity.x;
		d_pixels[pixelIndex].position.y += dt * d_pixels[pixelIndex].velocity.y;


	}
}


