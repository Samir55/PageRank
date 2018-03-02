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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		for (int i = 0; i < n; i++) {
			c_element += (d_a[idx * n + i] * d_b[i]);
		}
		d_c[idx] = c_element;
	}

	// Sync threads to update the d_b vector
	syncthreads();

	// copy the resulted vector from c to b for re multiplying and final result of course
	// after the final iteration will be stored in d_c
	d_b[idx] = d_c[idx];
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
	int matirx_bytes = sizeof(double) * n * n;
	int vector_bytes = sizeof(double) * n;

	// Allocate memory at the device for matrices a, b, and the result c
	cudaMalloc((void **) &d_a, matirx_bytes);
	cudaMalloc((void **) &d_b, vector_bytes);
	cudaMalloc((void **) &d_c, vector_bytes);

	// Copy matrices a & b to the device
	cudaMemcpy(d_a, h_a, matirx_bytes, cudaMemcpyHostToDevice);
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (A): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}
	cudaMemcpy(d_b, h_b, vector_bytes, cudaMemcpyHostToDevice);
	e =cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (B): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}

}

Matrix Kernel::get_result() {
	Matrix h_c = new double[n];

	int vector_bytes = sizeof(double) * n;

	cudaMemcpy(h_c, d_c, vector_bytes, cudaMemcpyDeviceToHost);
	cudaError_t e=cudaGetLastError();
	if(e!=cudaSuccess) {
		printf("MemCpy (R): CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}

	return h_c;
}

} /* namespace PageRank */
