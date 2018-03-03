#include "Kernel.h"
#include "GraphReader.h"
#include <iomanip>
using namespace std;
using namespace PageRank;

int main(int argc, char **argv)
{
    /* Debugging for read graph
    cout << nodesList.size() << endl;
    int i = 0;
    for(auto node : nodesList) {
        cout << "Node " << i++ << endl;
        cout << "In degrees " << node.size() << endl;
        cout << "Point to it nodes are ";
        for (auto p : node) {
            cout << p << " ";
        }
        cout << "\n ===============================================================" << endl;
    }
    */

    string path = "/home/ahmedsamir/SearchEngineRanker/Input/input.txt";
    vector< vector<int> > nodesList = GraphReader::read_graph(path);

    Matrix g_matrix, i_vector;
    int* out_degrees;

    // Construct S = H + A matrix
    GraphReader::construct_h_matrix(nodesList, g_matrix, i_vector, out_degrees);
    GraphReader::free_resources();

    Kernel page_rank(100, int(nodesList.size()));
    page_rank.allocate_matrices(g_matrix, i_vector);
    page_rank.run_kernel();

//    for (int i = 0;i < int(nodesList.size()) * int(nodesList.size()) ; i++) {
//        if (i && i%int(nodesList.size()) == 0) cout << endl;
//        cout << g_matrix[i] << "       ";
//    }
//    cout << endl;

	double *res = page_rank.get_result(), check_sum = 0.0;
	for (int i = 0; i < int(nodesList.size()) ; i++) {
		cout << i << " = " << setprecision(20) << res[i] << endl;
        check_sum += res[i];
	}
	cout << endl;
cout << "CHECK SUM " << check_sum << endl;

	return 0;
}

