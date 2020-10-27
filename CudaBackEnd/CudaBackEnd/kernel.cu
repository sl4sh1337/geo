
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


extern "C" void kern(uint32_t * data, uint32_t w, uint32_t h, uint32_t * ans, int* kernel)
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


__global__ void intersum(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
						uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					*ansij = *ansij | (min(((*dij >> (k * 8)) & 0xff) + ((*d1ij >> (k * 8)) & 0xff), 0xff) << (k * 8));
				}
			}

		}
	}
}

__global__ void intersub(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					*ansij = *ansij | (max((int)((*dij >> (k * 8)) & 0xff) - (int)((*d1ij >> (k * 8)) & 0xff), 0) << (k * 8));
				}
			}

		}
	}
}

__global__ void intermult(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					*ansij = *ansij | (min((int)((*dij >> (k * 8)) & 0xff) * (int)((*d1ij >> (k * 8)) & 0xff), 0xff) << (k * 8));
				}
			}

		}
	}
}


__global__ void interdiv(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					*ansij = *ansij | (max((int)((*dij >> (k * 8)) & 0xff) / (int)((*d1ij >> (k * 8)) & 0xff), 0) << (k * 8));
				}
			}

		}
	}
}

__global__ void interless(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if((int)((*dij >> (k * 8)) & 0xff) < (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void interlesseq(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if ((int)((*dij >> (k * 8)) & 0xff) <= (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void intergt(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if ((int)((*dij >> (k * 8)) & 0xff) > (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void intergteq(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if ((int)((*dij >> (k * 8)) & 0xff) >= (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void intereq(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if ((int)((*dij >> (k * 8)) & 0xff) == (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void interneq(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				for (size_t k = 0; k < 3; k++)
				{
					if ((int)((*dij >> (k * 8)) & 0xff) != (int)((*d1ij >> (k * 8)) & 0xff))
						*ansij = *ansij | (0xff << (k * 8));
					else
						*ansij = *ansij | (0 << (k * 8));
				}
			}

		}
	}
}

__global__ void interand(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				*ansij = 0xff000000 | ((0x00ffffff & *dij) & (0x00ffffff & *d1ij));
			}

		}
	}
}

__global__ void interor(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				*ansij = 0xff000000 | ((0x00ffffff & *dij) | (0x00ffffff & *d1ij));
			}

		}
	}
}

__global__ void interxor(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				*ansij = 0xff000000 | ((0x00ffffff & *dij) ^ (0x00ffffff & *d1ij));
			}

		}
	}
}

__global__ void interequiv(uint32_t* ans, size_t pitchans, uint32_t* data, uint32_t* data1, size_t pitchdata, size_t pitchdata1,
	uint32_t w, uint32_t h, uint32_t data_x0, uint32_t data_y0, uint32_t data_x1, uint32_t data_y1, uint32_t w1, uint32_t h1, uint32_t data1_x0, uint32_t data1_y0, uint32_t data1_x1, uint32_t data1_y1)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if ((i < data_y0 || i > data_y1) || (j < data_x0 || j > data_x1))
		{
			*ansij = *dij;
		}
		else
		{
			if (i + data1_y0 < h1 && j + data1_x0 < w1)
			{
				*ansij = 0xff000000;
				uint32_t* d1ij = (uint32_t*)((char*)data1 + (i + data1_y0) * pitchdata) + (j + data1_x0);
				*ansij = 0xff000000 | ~((0x00ffffff & *dij) ^ (0x00ffffff & *d1ij));
			}

		}
	}
}

extern "C" void intersec(uint32_t * data, uint32_t data_x0, uint32_t data_x1, uint32_t data_y0, uint32_t data_y1, uint32_t w, uint32_t h,
						uint32_t * ans,
						uint32_t * data1, uint32_t data1_x0, uint32_t data1_x1, uint32_t data1_y0, uint32_t data1_y1, uint32_t w1, uint32_t h1,
						int type)
{
	dim3 block_size(32, 32, 1);
	dim3 grid_size((w + 32 - 1) / 32, (h + 32 - 1) / 32, 1);

	size_t pitchdata, pitchdata1, pitchans;

	uint32_t* ddata; uint32_t* ddata1; uint32_t* dans;

	cudaMallocPitch(&ddata, &pitchdata, sizeof(int) * w, h);
	cudaMallocPitch(&ddata1, &pitchdata1, sizeof(int) * w1, h1);
	cudaMallocPitch(&dans, &pitchans, sizeof(int) * (w), h);

	cudaMemcpy2D(ddata, pitchdata, data, sizeof(int) * w, sizeof(int) * w, h, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy2D(ddata1, pitchdata1, data1, sizeof(int) * w1, sizeof(int) * w1, h1, cudaMemcpyKind::cudaMemcpyHostToDevice);

	switch (type)
	{
	case 1:
		intersum << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 2:
		intersub << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 3:
		intermult << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 4:
		interdiv << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 5:
		interless << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 6:
		interlesseq << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 7:
		intergt << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 8:
		intergteq << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 9:
		intereq << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 10:
		interneq << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 11:
		interand << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 12:
		interor << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 13:
		interxor << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	case 14:
		interequiv << <grid_size, block_size >> > (dans, pitchans, ddata, ddata1, pitchdata, pitchdata1, w, h, data_x0, data_y0, data_x1, data_y1, w1, h1, data1_x0, data1_y0, data1_x1, data1_y1);
		break;
	default:
		break;
	}
	

	cudaMemcpy2D(ans, sizeof(int) * (w), dans, pitchans, sizeof(int) * (w), h, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(dans);
	cudaFree(ddata);
	cudaFree(ddata1);
}



__global__ void nearest(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h, double offset_i, double offset_j)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		int di = i * offset_i, dj = j * offset_j;
		uint32_t* dij = (uint32_t*)((char*)data + di * pitchdata) + dj;
		*ansij = *dij;
	}
}


__global__ void bilin(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h, uint32_t ans_w, uint32_t ans_h, double offset_i, double offset_j)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < ans_h && j < ans_w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		int j0 = j / offset_j, int j1 = ((j / offset_j) + 1 >= w) ? (w - 1) : ((j / offset_j) + 1);
		int i0 = i / offset_i, i1 = ((i / offset_i) + 1 >= h) ? (h - 1) : ((i / offset_i) + 1);
		if (i % (int)offset_i == 0 && j % (int)offset_j == 0)
		{
			*ansij = *((uint32_t*)((char*)data + (int)(i / offset_i) * pitchdata) + (int)(j / offset_j));
		}
		else
		{
			*ansij = 0xff000000;
			for (size_t k = 0; k < 3; k++)
			{
				uint32_t q00 = ((*((uint32_t*)((char*)data + i0 * pitchdata) + j0)) >> (k * 8)) & 0xff;
				uint32_t q01 = ((*((uint32_t*)((char*)data + i1 * pitchdata) + j0)) >> (k * 8)) & 0xff;
				uint32_t q10 = ((*((uint32_t*)((char*)data + i0 * pitchdata) + j1)) >> (k * 8)) & 0xff;
				uint32_t q11 = ((*((uint32_t*)((char*)data + i1 * pitchdata) + j1)) >> (k * 8)) & 0xff;

				double x0 = 0.0, x1 = 1.0, x = (double)(j - j0 * offset_j) / offset_j;
				double y0 = 0.0, y1 = 1.0, y = (double)(i - i0 * offset_i) / offset_i;

				double fr0 = ((x1 - x) / (x1 - x0)) * q00 + ((x - x0) / (x1 - x0)) * q10;
				double fr1 = ((x1 - x) / (x1 - x0)) * q01 + ((x - x0) / (x1 - x0)) * q11;

				uint32_t fp = (uint32_t)(((y1 - y) / (y1 - y0)) * fr0 + ((y - y0) / (y1 - y0)) * fr1);
				if (fp < 0)
					*ansij = 0 | *ansij;
				else if (fp > 255)
					*ansij = (255 << (k * 8)) | *ansij;
				else
					*ansij = (fp << (k * 8)) | *ansij;
			}
		}
	}
}

extern "C" void resampling(uint32_t * data, uint32_t w, uint32_t h, uint32_t * ans, uint32_t ans_w, uint32_t ans_h, double q)
{
	dim3 block_size(16, 16, 1);
	dim3 grid_size((ans_w + 16 - 1) / 16, (ans_h + 16 - 1) / 16, 1);

	uint32_t offset_j, offset_i;

	if(w < ans_w)
		offset_j = ans_w / w, offset_i = ans_h / h;
	else
		offset_j = w / ans_w, offset_i = h / ans_h;

	size_t pitchdata, pitchans;

	uint32_t* ddata; uint32_t* dans;

	cudaMallocPitch(&ddata, &pitchdata, sizeof(int) * w, h);
	cudaMallocPitch(&dans, &pitchans, sizeof(int) * ans_w, ans_h);

	cudaMemcpy2D(ddata, pitchdata, data, sizeof(int) * w, sizeof(int) * w, h, cudaMemcpyKind::cudaMemcpyHostToDevice);

	if(w < ans_w)
		bilin <<<grid_size, block_size >>>(dans, pitchans, ddata, pitchdata, w, h, ans_w, ans_h, q, q);
	else
		nearest <<<grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, ans_w, ans_h, 1.0 / q, 1.0 / q);
	cudaMemcpy2D(ans, sizeof(int) * ans_w, dans, pitchans, sizeof(int) * ans_w, ans_h, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(dans);
	cudaFree(ddata);
}

__global__ void cuabs(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000;
		for (size_t k = 0; k < 3; k++)
		{
			if (((*dij >> (k * 8)) & 0xff) < 0x80)
				*ansij = *ansij | (0x00 << (k * 8));
			else
				*ansij = *ansij | (0xff << (k * 8));
		}
	}
}

__global__ void curound(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000;
		for (size_t k = 0; k < 3; k++)
		{
			if (((*dij >> (k * 8)) & 0xff) < 0x40)
				*ansij = *ansij | 0;
			else if(((*dij >> (k * 8)) & 0xff) < 0x80)
				*ansij = *ansij | (0x80 << (k * 8));
			else if (((*dij >> (k * 8)) & 0xff) < 0xc0)
				*ansij = *ansij | (0x80 << (k * 8));
			else
				*ansij = *ansij | (0xff << (k * 8));
		}
	}
}

__global__ void cufloor(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000;
		for (size_t k = 0; k < 3; k++)
		{
			if (((*dij >> (k * 8)) & 0xff) < 0x80)
				*ansij = *ansij | (0x00 << (k * 8));
			else
				*ansij = *ansij | (0x80 << (k * 8));
		}
	}
}

__global__ void cuceil(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000;
		for (size_t k = 0; k < 3; k++)
		{
			if (((*dij >> (k * 8)) & 0xff) < 0x80)
				*ansij = *ansij | (0x80 << (k * 8));
			else
				*ansij = *ansij | (0xff << (k * 8));
		}
	}
}


__global__ void cusqrt(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000 | ((int)sqrtf(0x00ffffff & *dij));
	}
}

__global__ void culog(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000 | ((int)logf(0x00ffffff & *dij));
	}
}

__global__ void cuexp(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		*ansij = 0xff000000 | (max((int)expf(0x00ffffff & *dij), 0x00ffffff));
	}
}

__global__ void cucos(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if(cosf(0x00ffffff & *dij) <= 0)
			*ansij = 0xff000000;
		else
			*ansij = 0xffffffff;
	}
}

__global__ void cusin(uint32_t* ans, size_t pitchans, uint32_t* data, size_t pitchdata, uint32_t w, uint32_t h)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y, j = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < h && j < w)
	{
		uint32_t* ansij = (uint32_t*)((char*)ans + i * pitchans) + j;
		uint32_t* dij = (uint32_t*)((char*)data + i * pitchdata) + j;
		if (sinf(0x00ffffff & *dij) <= 0)
			*ansij = 0xff000000;
		else
			*ansij = 0xffffffff;
	}
}


extern "C" void singular(uint32_t * data, uint32_t w, uint32_t h, uint32_t * ans, int func)
{
	dim3 block_size(16, 16, 1);
	dim3 grid_size((w + 16 - 1) / 16, (h + 16 - 1) / 16, 1);

	//double offset_j = (double)w / ans_w, offset_i = (double)h / ans_h;

	size_t pitchdata, pitchans;

	uint32_t* ddata; uint32_t* dans;

	cudaMallocPitch(&ddata, &pitchdata, sizeof(int) * w, h);
	cudaMallocPitch(&dans, &pitchans, sizeof(int) * w, h);

	cudaMemcpy2D(ddata, pitchdata, data, sizeof(int) * w, sizeof(int) * w, h, cudaMemcpyKind::cudaMemcpyHostToDevice);

	switch (func)
	{
	case 1:
		cuabs <<<grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 2:
		curound << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 3:
		cufloor << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 4:
		cuceil << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 5:
		cusqrt << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 6:
		culog << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 7:
		cuexp << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 8:
		cucos << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	case 9:
		cusin << <grid_size, block_size >> > (dans, pitchans, ddata, pitchdata, w, h);
		break;
	default:
		break;
	}
		

	cudaMemcpy2D(ans, sizeof(int) * w, dans, pitchans, sizeof(int) * w, h, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(dans);
	cudaFree(ddata);
}