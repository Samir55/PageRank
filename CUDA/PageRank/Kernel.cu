/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#include "Kernel.h"

namespace PageRank {

// A Kernel for multiplying square matrices.
__global__ void matrix_mul(Matrix d_a, Matrix d_b, Matrix d_c) {
	double c_element = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < m) {
	for (int i = 0; i < n; i++) {
		c_element += (d_a[row * n + i] * d_b[i * n + col]);
	}
	d_c[row * n + col] = c_element;
}
}

Kernel::Kernel() {
}

Kernel::~Kernel() {
}

void Kernel::run_kernel(Matrix d_a, Matrix d_b, Matrix d_c) { 
	for (int i = 0; i < max_iterations; ++i)
	{
		matrix_mul<<<gridSize, blockSize>>>(d_a, d_b, d_c);
	}
}

void allocate_memory() {
	
}

} /* namespace PageRank */
