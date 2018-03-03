/*
 * GraphReader.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#include "GraphReader.h"

namespace PageRank {

    int *GraphReader::out_degrees = NULL;
    int GraphReader::total_edges = 0;

    GraphReader::GraphReader() {}

    GraphReader::~GraphReader() {}

    vector<vector<int> > GraphReader::read_graph(string path) {
        ifstream file;
        file.open(path.c_str());

        int n, to, from;
        file >> n;
        vector<vector<int> > edges_list(n);
        GraphReader::out_degrees = new int[n];

        for (int i = 0; i < n; ++i) {
            out_degrees[i] = 0;
        }

        while (file >> from >> to) {
            edges_list[to].push_back(from);
            out_degrees[from]++;
            total_edges++;
        }

        return edges_list;
    }

    int GraphReader::construct_h_matrix(vector<vector<int> > &edges_list, int* &h_g, I &h_i,
                                           page* &h_pages) {
        out_degrees = GraphReader::out_degrees;
        int n = int(edges_list.size());

        h_g = new int[total_edges];
        h_i = new double[n];
        h_pages = new page[n];


        // Initialize the I vector with 1/n values
        for (int i = 0; i < n; ++i) {
            h_i[i] = 1.0 / n;
        }

        int next_idx = 0;
        for (int i = 0; i < n; ++i) {
            if (out_degrees[i] > 0) {
                h_pages[i].out_degrees = out_degrees[i];
                h_pages[i].dangling = false;
                h_pages[i].start_idx = next_idx;
                h_pages[i].end_idx =  int(edges_list[i].size()) + next_idx;
                for (auto from : edges_list[i]) {
                    h_g[next_idx++] = from;
                }
            }
        }

        return GraphReader::total_edges;
    }

    void GraphReader::free_resources() {
        // free the resources
        delete out_degrees;
    }


} /* namespace PageRank */
