/*
 * GraphReader.h
 *
 *  Created on: Feb 27, 2018
 *      Author: ahmedsamir
 */

#ifndef GRAPHREADER_H_
#define GRAPHREADER_H_

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <iomanip>
#include "utils.h"

using namespace std;
typedef double* I;

namespace PageRank {

class GraphReader {
public:
	GraphReader();

	static int *out_degrees;
	static int total_edges;

	static vector< vector<int> > read_graph(string path);

	static int construct_h_matrix(vector<vector<int> > &edges_list, int* &h_g, I &h_i, page* &h_page);

	static void free_resources();

	virtual ~GraphReader();
};

} /* namespace PageRank */

#endif /* GRAPHREADER_H_ */
