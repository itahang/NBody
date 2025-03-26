#include "GPUpart.cuh"
#include <cuda_runtime.h>

__device__ float2 normalizedVector(float2 vec, float epslon = 0.0001) {
	float dist_sq = vec.x * vec.x + vec.y * vec.y + epslon;
	float inv_dist = rsqrtf(dist_sq);
	return { vec.x * inv_dist, vec.y * inv_dist };
}

__device__ float2 calcAcceleration(Body* s, Body* t, float epsilon, float rfactor = 100.0) {
	float2 r = { rfactor * (t->position.x - s->position.x),rfactor * (t->position.y - s->position.y) };

	float rfac = powf(r.x * r.x + r.y * r.y, -1.5);
	float mass = s->mass * t->mass;

	return normalizedVector({ mass * rfac * r.x,mass * rfac * r.y });
}


__device__ float2 meanposition;

__global__ void kernel(Body* d_pixels, int Width, int Height) {


	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	const float dt = 0.001;
	const float epsilon = 1e-2;

	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;

		if (pixelIndex == 0) {
			meanposition = { 0,0 };
		}

		float2 old_position = d_pixels[pixelIndex].position;

		d_pixels[pixelIndex].acceleration = { 0, 0 };

		for (int i = 0; i < Width * Height; i++) {
			if (i != pixelIndex) {
				float2 tAcc = calcAcceleration(&d_pixels[pixelIndex], &d_pixels[i], epsilon);
				d_pixels[pixelIndex].acceleration.x += tAcc.x;
				d_pixels[pixelIndex].acceleration.y += tAcc.y;
			}
		}

		float2 new_position;
		new_position.x = 2 * old_position.x - d_pixels[pixelIndex].prev_position.x +
			dt * dt * d_pixels[pixelIndex].acceleration.x;
		new_position.y = 2 * old_position.y - d_pixels[pixelIndex].prev_position.y +
			dt * dt * d_pixels[pixelIndex].acceleration.y;

		d_pixels[pixelIndex].prev_position = old_position;
		d_pixels[pixelIndex].position = new_position;

		float tempx = d_pixels[pixelIndex].position.x / (Width * Height);
		float tempy = d_pixels[pixelIndex].position.y / (Width * Height);;
		atomicAdd(&meanposition.x, tempx);
		atomicAdd(&meanposition.y, tempy);
	}
}


__global__ void changeMean(Body* d_pixels, int Width, int Height) {
	// Taking Average mean;

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	const float dt = 0.001;
	const float epsilon = 1e-2;

	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;
		d_pixels[pixelIndex].position.x -= meanposition.x;
		d_pixels[pixelIndex].position.y -= meanposition.y;

	}
}