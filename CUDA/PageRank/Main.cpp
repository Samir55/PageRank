#include "GPUTimer.h"
#include "Kernel.h"
#include "GraphReader.h"

using namespace PageRank;

int main(int argc, char **argv) {
    if (argc <= 1) return 0;

    string path = argv[1];

    page* h_pages;
    int* h_g;
    I h_i;
    int nodes_count;

    GPUTimer gpu_timer;

    // Read graph file.
    vector<vector<int> > nodesList = GraphReader::read_graph(path);
    nodes_count = int(nodesList.size());

    // Construct S = H + A matrix
    int total_edges = GraphReader::construct_h_matrix(nodesList, h_g, h_i, h_pages);

    // Free resources.
    GraphReader::free_resources();

    // Create page rank kernel object
    Kernel page_rank(nodes_count);

    // Allocate matrices in the gpu memory
    page_rank.allocate_matrices(total_edges, h_g, h_i, h_pages);

    // Run PageRank algorithm
    gpu_timer.start();

    page_rank.run_kernel();

    gpu_timer.stop();

    // Save Result in output.txt file
    ofstream file;
    file.open("output.txt");

    double *res = page_rank.get_result(), check_sum = 0.0;
    for (int i = 0; i < int(nodesList.size()); i++) {
        file << i << " = " << setprecision(20) << res[i] << endl;
        check_sum += res[i];
    }

    // Print Elapsed time
    cout << "Elapsed PageRank time in gpu " << gpu_timer.elapsed() << " ms" << endl;
    return 0;
}

