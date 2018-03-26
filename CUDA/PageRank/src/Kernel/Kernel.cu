/*
 * Kernel.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Ahmed Samir
 */

#include "Kernel.hpp"

namespace PageRank {

// A Kernel for multiplying square matrices.
    __global__ void page_rank_iteration (Page* d_pages, double* d_page_probs, int* d_edges_list, int pages_count, double d_dangling_sum, double alpha) {
        int idx = blockIdx.y * blockDim.y + threadIdx.y;

        double c_element = 0.0;
        int i_start = d_pages[idx].start_idx;
        int i_end = d_pages[idx].start_idx;

        for (int i = i_start; i < i_end; i++) {
            // Get the index of current node linking to this node
            int from = d_edges_list[i];

            c_element += d_page_probs[from] * 1 / d_pages[from].out_links_count; 
        }

        c_element = (1 - alpha) * 1.0 / pages_count + alpha * (c_element + d_dangling_sum / pages_count);

        __syncthreads();

        d_page_probs[idx] = c_element;
    }

    void Kernel::run_kernel(int dangling_nodes_count) {
        // Calculate the grid and block sizes.
        int grid_size = int(ceil(1.0 * pages_count / MAX_BLOCK_SIZE));
        int block_size = int(ceil(1.0 * pages_count / grid_size));

        if (block_size < 1024) {
            dim3 dimGrid(1, grid_size);
            dim3 dimBlock(1, block_size);

            for (int i = 0; i < MAX_ITERATIONS; ++i) {
                page_rank_iteration << < dimGrid, dimBlock >> > (d_pages, d_pages_probs, d_edges_list, pages_count, dangling_nodes_count, ALPHA);
            }
        } else {
            cout << "Error exceeded the maximum value for threads in a block 1024" << endl;
        }
    }

    void Kernel::allocate_data(Page* h_pages, double* h_pages_probs, int* h_edges_list) {
        // Allocate memory at the gpu device
        cudaMalloc ((void **) &d_pages, sizeof(Page) * pages_count);
        cudaMalloc ((void **) &d_pages_probs, sizeof(double) * pages_count);
        cudaMalloc ((void **) &d_edges_list, sizeof(int) * edges_count);

        // Copy data from host (cpu) to the gpu
        cudaMemcpy(d_pages, h_pages, sizeof(Page) * pages_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_pages_probs, h_pages_probs, sizeof(double) * pages_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_edges_list, h_edges_list, sizeof(int) * edges_count, cudaMemcpyDeviceToHost);
    }

    double* Kernel::get_result() {
        double* pages_probs = new double[pages_count];

        cudaMemcpy(pages_probs, d_pages_probs, sizeof(double) * pages_count, cudaMemcpyDeviceToHost);
               cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) {
                    printf("MemCpy (R): CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
                    exit(0);
                }

        return pages_probs;
    }

} /* namespace PageRank */
