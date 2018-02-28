/*
 * Kernel.h
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;

#define GRID_COLS 200
#define GRID_ROWS 200
#define BLOCK_ROWS 25
#define BLOCK_COLS 25

typedef double* Matrix;

namespace PageRank {

class Kernel {
private:
	int max_iterations;
	int n;
	Matrix d_a;
	Matrix d_b;
	Matrix d_c;

public:
	Kernel(int iterations, int n) : max_iterations(iterations), n(n) {}
	virtual ~Kernel() {}

	void allocate_matrices(Matrix h_a, Matrix h_b);
	void run_kernel();
	Matrix get_result();
};

} /* namespace PageRank */

#endif /* KERNEL_H_ */
