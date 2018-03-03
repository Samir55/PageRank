/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Ahmed Samir
 */

#include "Kernel.h"

namespace PageRank {

// A Kernel for multiplying square matrices.
    __global__ void page_rank_iteration(int* d_g, double* d_i, double* d_tmp, page* d_pages, d_dangling_sum d_ranks_sum, int n, double alpha) {
        double c_element = 0.0;

        int idx = blockIdx.y * blockDim.y + threadIdx.y;

    }

    __global__ void update_i_vector(double* d_i, double* d_tmp,) {

    }

    void Kernel::run_kernel() {
        // Calculate the grid and block sizes.
        int grid_size = int(ceil(1.0 * n / MAX_BLOCK_SIZE));
        int block_size = int(ceil(1.0 * n / grid_size));

        if (block_size < 1024) {
            dim3 dimGrid(1, grid_size);
            dim3 dimBlock(1, block_size);

            for (int i = 0; i < MAX_ITERATIONS; ++i) {
                page_rank_iteration << < dimGrid, dimBlock >> > (d_g, d_i, d_tmp, d_pages, d_dangling_sum, d_ranks_sum, n, ALPHA);
                update_i_vector << < dimGrid, dimBlock >> > (d_i, d_tmp);
            }
        } else {
            cout << "Error exceeded the maximum value for threads in a block 1024" << endl;
        }
    }

    void Kernel::allocate_matrices(int total_edges, int *h_g, I h_i, page *h_pages) {
        long h_g_size = sizeof(int) * total_edges;
        long h_page_size = sizeof(page) * n;
        long h_i_size = sizeof(double) * n;
        double h_ranks_size = sizeof(double);
        double h_dangling_size = sizeof(double);

        double h_ranks_sum = 1.0;
        double h_dangling_sum = 0.0;

        // Allocate memory at the device for matrices a, b, and the result c
        cudaMalloc((void **) &d_g, h_g_size);
        cudaMalloc((void **) &d_pages, h_page_size);
        cudaMalloc((void **) &d_i, h_i_size);
        cudaMalloc((void **) &d_tmp, h_i_size);
        cudaMalloc((void **) &d_ranks_sum, h_ranks_size);
        cudaMalloc((void **) &d_dangling_sum, h_dangling_size);


        // Copy matrices a & b to the device
        cudaMemcpy(d_g, h_g, h_g_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pages, h_pages, h_page_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, h_i, h_i_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ranks_sum, &h_ranks_sum, h_ranks_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dangling_sum, h_dangling_sum, h_dangling_size, cudaMemcpyHostToDevice);

    }

    Matrix Kernel::get_result() {
//
//        Matrix h_c = new double[n];
//
//        int vector_bytes = sizeof(double) * n;
//
//        cudaMemcpy(h_c, d_c, vector_bytes, cudaMemcpyDeviceToHost);
//        cudaError_t e = cudaGetLastError();
//        if (e != cudaSuccess) {
//            printf("MemCpy (R): CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
//            exit(0);
//        }

        return 0;
    }

} /* namespace PageRank */
