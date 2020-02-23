
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>


__global__ void convolution(uint32_t* ans, size_t pitchans, uint32_t* data, int* kernel, size_t pitchdata, size_t pitchkernel, uint32_t w, uint32_t h, int kern_sum)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		*ansij = 0xff000000;
		for (size_t k = 0; k < 3; k++)
		{
			int s = 0;
			for (int m = 0; m < 3; ++m)
			{
				for (int n = 0; n < 3; ++n)
				{
					s += ((*((int*)((char*)data + (i + (2 - m)) * pitchdata) + j + (2 - n)) >> (k * 8)) & 0xff)** ((int*)((char*)kernel + m * pitchkernel) + n);
				}
			}
			if (s / kern_sum < 0)
				*ansij = 0 | *ansij;
			else if (s / kern_sum > 255)
				*ansij = (255 << (k * 8)) | *ansij;
			else
				*ansij = (s / kern_sum << (k * 8)) | *ansij;
		}

	}
}


extern "C" void kern(uint32_t * data, uint32_t h, uint32_t w, uint32_t * ans, int* kernel)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size(((w - 2) + 32 - 1) / 32, ((h - 2) + 32 - 1) / 32, 1);


	int kern_sum = 0;
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			kern_sum += kernel[i * 3 + j];
		}
	}
	if (kern_sum == 0)
		kern_sum = 1;

	size_t pitchdata, pitchkernel, pitchans;

	uint32_t* ddata; int* dkernel; uint32_t* dans;

	cudaMallocPitch(&ddata, &pitchdata, sizeof(int) * w, h);
	cudaMallocPitch(&dkernel, &pitchkernel, sizeof(int) * 3, 3);
	cudaMallocPitch(&dans, &pitchans, sizeof(int) * (w - 2), h - 2);

	cudaMemcpy2D(ddata, pitchdata, data, sizeof(int) * w, sizeof(int) * w, h, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy2D(dkernel, pitchkernel, kernel, sizeof(int) * 3, sizeof(int) * 3, 3, cudaMemcpyKind::cudaMemcpyHostToDevice);


	convolution << <grid_size, block_size >> > (dans, pitchans, ddata, dkernel, pitchdata, pitchkernel, w - 2, h - 2, kern_sum);

	cudaMemcpy2D(ans, sizeof(int) * (w - 2), dans, pitchans, sizeof(int) * (w - 2), h - 2, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(dans);
	cudaFree(ddata);
	cudaFree(dkernel);
}