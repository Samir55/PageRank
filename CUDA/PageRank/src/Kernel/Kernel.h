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
#include "utils.h"

using namespace std;

#define MAX_ITERATIONS 100
#define ALPHA 0.85
#define MAX_BLOCK_SIZE 16


typedef double* I;

namespace PageRank {

class Kernel {
private:

	int n;
    int* d_g;
    page* d_pages;
    I d_i;
    I d_tmp;
    double* d_ranks_sum;
    double* d_dangling_sum;

public:
	Kernel(int n) : n(n) {}
	virtual ~Kernel() {}

	void allocate_matrices(int total_edges, int* h_g, I h_i, page* h_pages);

	void run_kernel();

	double* get_result();
};

} /* namespace PageRank */

#endif /* KERNEL_H_ */
