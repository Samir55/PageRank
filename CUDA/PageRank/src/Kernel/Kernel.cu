/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Ahmed Samir
 */

#include "Kernel.hpp"

namespace PageRank {

// A Kernel for multiplying square matrices.
    __global__ void page_rank_iteration(int* d_g, double* d_i, double* d_tmp, page* d_pages, d_dangling_sum d_ranks_sum, int n, double alpha) {
        double c_element = 0.0;

        int idx = blockIdx.y * blockDim.y + threadIdx.y;

    }

    __global__ void update_i_vector(double* d_i, double* d_tmp,) {

    }

    void Kernel::run_kernel(int dangling_nodes_count) {
        // Calculate the grid and block sizes.
        int grid_size = int(ceil(1.0 * n / MAX_BLOCK_SIZE));
        int block_size = int(ceil(1.0 * n / grid_size));

        if (block_size < 1024) {
            dim3 dimGrid(1, grid_size);
            dim3 dimBlock(1, block_size);

            for (int i = 0; i < MAX_ITERATIONS; ++i) {
                page_rank_iteration << < dimGrid, dimBlock >> > (d_g, d_i, d_tmp, d_pages, d_dangling_sum, d_ranks_sum, n, ALPHA);
            }
        } else {
            cout << "Error exceeded the maximum value for threads in a block 1024" << endl;
        }
    }

    void Kernel::allocate_data(Page* h_pages, double* h_pages_probs, int* h_edges_list) {
        // Allocate memory at the gpu device
        cudaMalloc ((void **) &d_pages, sizeof(Page) * pages_count);
        cudaMalloc ((void **) &d_page_probs, sizeof(double) * pages_count);
        cudaMalloc ((void **) &d_edges_list, sizeof(int) * edges_count);

        // Copy data from host (cpu) to the gpu
        cudaMemcpy(d_pages, h_pages, sizeof(Page) * pages_count);
        cudaMemcpy(d_page_probs, h_pages_probs, sizeof(double) * pages_count);
        cudaMemcpy(d_edges_list, h_edges_list, sizeof(int) * edges_count);
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
