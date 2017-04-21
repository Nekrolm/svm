#include "../inc/Matrix.h"
#include "../inc/SVM.h"

using namespace std;
using namespace Classificator;
using namespace Numeric;

int main(){
	freopen("input.txt", "r", stdin);
	freopen("log1.txt","w", stderr);
	size_t N,D,C, steps;
	double lmbd;

	cin >> N >> D >> C >> lmbd >> steps;

	SVM classificator(Matrix::get_matrix_from(cin, {C, D+1}), lmbd, steps);

	Matrix X = Matrix::get_matrix_from(
			cin, {D , N}
		).append(Matrix(1,N,1)).reshape({D+1, N});

	double loss = classificator.fit(X, Matrix::get_matrix_from(cin, {1, N}));

	auto Z = Matrix::get_matrix_from(
			cin, {D, 3}
		).append(Matrix(1,3,1)).reshape({D+1, 3});

	for (double c : classificator.predict(Z))
		cout << c << " ";

	cout << endl << loss;
	return EXIT_SUCCESS;
}



