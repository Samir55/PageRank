/*
 * Kernel.h
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#ifndef KERNEL_H_
#define KERNEL_H_

namespace PageRank {

typedef int* Matrix;

class Kernel {
public:
	int const max_iterations;
	int const n = 5000;
	int const m = 5000;
	// gridSize 200  200
	// blockSize 25 25 
	Kernel();
	virtual ~Kernel();

	static void run_kernel() {}
};

} /* namespace PageRank */

#endif /* KERNEL_H_ */
