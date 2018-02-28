/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Ahmed Samir
 */

#include "Kernel.h"

namespace PageRank {

// A Kernel for multiplying square matrices.
__global__ void matrix_mul(Matrix d_a, Matrix d_b, Matrix d_c, int n) {
	double c_element = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		for (int i = 0; i < n; i++) {
			c_element += (d_a[row * n + i] * d_b[i * n + col]);
		}
		d_c[row * n + col] = c_element;
	}
}

void Kernel::run_kernel() {
	dim3 dimGrid(GRID_ROWS,GRID_COLS);
	dim3 dimBlock(BLOCK_ROWS, BLOCK_COLS);

	for (int i = 0; i < max_iterations; ++i)
	{
		matrix_mul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
	}
}

void Kernel::allocate_matrices(Matrix h_a, Matrix h_b) {
	int array_bytes = sizeof(double) * n * n;

	// Allocate memory at the device for matrices a, b, and the result c
	cudaMalloc((void **) &d_a, array_bytes);
	cudaMalloc((void **) &d_b, array_bytes);
	cudaMalloc((void **) &d_c, array_bytes);

	// Copy matrices a & b to the device
	cudaMemcpy(d_a, h_a, array_bytes, cudaMemcpyHostToDevice);
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (A): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}
	cudaMemcpy(d_b, h_b, array_bytes, cudaMemcpyHostToDevice);
	e =cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (B): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}

}

Matrix Kernel::get_result() {
	Matrix h_c = new double[n*n];

	int array_bytes = sizeof(double) * n * n;

	cudaMemcpy(h_c, d_c, array_bytes, cudaMemcpyDeviceToHost);
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (R): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}

	return h_c;
}

} /* namespace PageRank */
