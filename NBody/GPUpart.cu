#include "GPUpart.cuh"
#include <cuda_runtime.h>

__device__ double2 normalizedVector(double2 vec, double epslon = 0.0001) {
	double dist_sq = vec.x * vec.x + vec.y * vec.y + epslon;
	double inv_dist = rsqrt(dist_sq);
	return { vec.x * inv_dist, vec.y * inv_dist };
}

__device__ double2 calcAcceleration(Body* s, Body* t, double epsilon, double rfactor = 100.0) {
	double2 r = { rfactor * (t->position.x - s->position.x),rfactor * (t->position.y - s->position.y) };

	double rfac = pow(r.x * r.x + r.y * r.y, -1.5);
	double mass = s->mass * t->mass;

	return normalizedVector({ mass * rfac * r.x,mass * rfac * r.y });
}


extern __device__ double2 meanposition;

__global__ void kernel(Body* d_pixels, int Width, int Height) {


	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	const double dt = 0.001;
	const double epsilon = 1e-2;

	if (idx < Width && idy < Height) {
		int pixelIndex = idx + Width * idy;

		meanposition = d_pixels[0].position;
		__syncthreads();

		double2 old_position = d_pixels[pixelIndex].position;

		d_pixels[pixelIndex].acceleration = { 0, 0 };

		for (int i = 0; i < Width * Height; i++) {
			if (i != pixelIndex) {
				double2 tAcc = calcAcceleration(&d_pixels[pixelIndex], &d_pixels[i], epsilon);
				d_pixels[pixelIndex].acceleration.x += tAcc.x;
				d_pixels[pixelIndex].acceleration.y += tAcc.y;
			}
		}

		double2 new_position;
		new_position.x = 2 * old_position.x - d_pixels[pixelIndex].prev_position.x +
			dt * dt * d_pixels[pixelIndex].acceleration.x;
		new_position.y = 2 * old_position.y - d_pixels[pixelIndex].prev_position.y +
			dt * dt * d_pixels[pixelIndex].acceleration.y;

		d_pixels[pixelIndex].prev_position = old_position;
		d_pixels[pixelIndex].position = new_position;

		d_pixels[pixelIndex].position.x = d_pixels[pixelIndex].position.x - meanposition.x;
		d_pixels[pixelIndex].position.y = d_pixels[pixelIndex].position.y - meanposition.y;

	}

}