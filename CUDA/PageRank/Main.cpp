#include "Kernel.h"
using namespace std;

int main(int argc, char **argv)
{
	PageRank::Kernel page_rank(1, 3);
	double h_a[] = {1,2,3,4,5,6,7,8,9};
	double h_b[] = {1,1,0,0,1,0,0,0,1};

	page_rank.allocate_matrices(h_a, h_b);
	page_rank.run_kernel();
	double *res = page_rank.get_result();
	for (int i = 0; i < 9 ; i++) {
		cout << res[i] << " ";
	}
	cout << endl;


	return 0;
}
