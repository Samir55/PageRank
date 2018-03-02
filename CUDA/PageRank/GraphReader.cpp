/*
 * GraphReader.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#include "GraphReader.h"

namespace PageRank {

int* GraphReader::out_degrees = NULL;

GraphReader::GraphReader() {}

GraphReader::~GraphReader() {}

vector< vector<int> > GraphReader::read_graph(string path) {
	ifstream file;
	file.open(path.c_str());

	int n, to, from;
	file >> n;
	vector< vector<int> > edges_list(n);
	int *out_degrees = new int[n];


	while (file >> to >> from) {
		edges_list[to].push_back(from);
		out_degrees[from]++;
	}

	return edges_list;
}

Matrix GraphReader::construct_h_matrix(vector< vector<int> > &edges_list, Matrix g_matrix, Matrix i_vector, int* out_degrees) {
	out_degrees = GraphReader::out_degrees;
	int n = int(edges_list.size());

	g_matrix = new double[n * n];
	i_vector = new double[n];


	// Initialize the I vector with 1/n values
	for (int i = 0; i < n; ++i) {
		i_vector[i] = 1.0 / n;
	}

	// Initialize the matrix
	memset(g_matrix, 0, sizeof(g_matrix));

	for (int i = 0; i < n; ++i) {

		if (out_degrees[i] > 0){
			for (int j = 0; j < edges_list[i].size() ; ++j) {
				g_matrix[i + n * edges_list[i][j]] = 1.0 / out_degrees[i];
			}
		} else {
			for (int row = 0; row < n; ++row) {
				g_matrix[i + n * row] = 1.0 / n;
			}
		}
	}
}

void GraphReader::free_resources() {
	// free the resources
	delete out_degrees;
}


} /* namespace PageRank */
