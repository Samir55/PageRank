/*
 * Kernel.h
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#define GRID_COLS 200
#define GRID_ROWS 200
#define BLOCK_ROWS 25
#define BLOCK_COLS 25

namespace PageRank {

typedef int* Matrix;

class Kernel {
public:
	static int const max_iterations;
	static int const n = 5000;
	static int const m = 5000;

	dim3 dimGrid(GRID_ROWS, GRID_COLS);
    dim3 dimBlock(BLOCK_ROWS, BLOCK_COLS);

	Kernel();
	virtual ~Kernel();

	static void allocate_memory();
	static void run_kernel() {}
};

} /* namespace PageRank */

#endif /* KERNEL_H_ */
