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
	Kernel();
	virtual ~Kernel();

	static void run_kernel() {}


};

} /* namespace PageRank */

#endif /* KERNEL_H_ */
