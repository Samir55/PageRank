#include "Kernel.h"
#include "GraphReader.h"
using namespace std;
using namespace PageRank;

int main(int argc, char **argv)
{
	Kernel page_rank(1, 3);

	double h_a[] = {1, 4, 7, 2, 4, 6, 3, 5, 9};
	double h_b[] = {0, 9, 10};

	page_rank.allocate_matrices(h_a, h_b);
	page_rank.run_kernel();
	double *res = page_rank.get_result();
	for (int i = 0; i < 3 ; i++) {
		cout << res[i] << " ";
	}
	cout << endl;


	return 0;
}

