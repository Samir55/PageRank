/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#include "Kernel.h"

namespace PageRank {
__global__ void matrix_mul(Matrix d_a, Matrix d_b, Matrix d_c) {
	double c_element = 0.0;
	int row = threadIdx.y + blockIdx.y;
	int col;
	}

Kernel::Kernel() {
	// TODO Auto-generated constructor stub

}

Kernel::~Kernel() {
	// TODO Auto-generated destructor stub
}

} /* namespace PageRank */
