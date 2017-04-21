#include "../inc/SVM.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <iomanip>
using namespace std;
using namespace Numeric;
namespace Classificator{

SVM::SVM(Matrix&& W0, double lambda, size_t steps) :
		W(forward<Matrix&&>(W0)), lambda(lambda), steps(steps)
{
	step.reserve(20);
	generate_n(back_inserter(step), 20, [](){
			double ret;
			static double s = 1e-7;
			ret = s;
			s+=2e-6;
			return ret;
		});
}

Matrix SVM::predict(const Matrix& x){
	auto scores = (W * x).transpose();
	Matrix ret = Matrix(1, x.get_shape().second);

	size_t C = scores.get_shape().second;

	for (size_t i = 0; i<x.get_shape().second; ++i)
		ret[i] = max_element(scores.begin() + i * C,
							scores.begin() + (i+1) * C) -
						(scores.begin() + i * C);

	return ret;
}

SVM::~SVM() {
	// TODO Auto-generated destructor stub
}


double SVM::loss(const Matrix& W, const Matrix& X, const Matrix& Y) const{
	Matrix tmp = W * X;
	double ret = 0;
	for (size_t i = 0; i < tmp.get_shape().second; ++i){
		size_t k = int(Y[i] + 0.5);
		for (size_t j = 0; j < tmp.get_shape().first; ++j)
			if (j != k)
				ret += max(0.0, tmp.at(j,i) - tmp.at(k, i) + 1);
	}
	return lambda * W.square_norm() + ret / X.get_shape().second;
}

Matrix SVM::grad_loss(const Matrix& W, const Matrix& X, const Matrix& Y) const{
	Matrix scores = W * X;
	Matrix grad(W.get_shape());


	for (size_t i = 0; i < scores.get_shape().second; ++i){
		size_t k = int(Y[i] + 0.5);
		for (size_t j = 0; j < scores.get_shape().first; ++j)
			if (j != k && scores.at(j,i) - scores.at(k,i) > -1){
				for (size_t p = 0; p < W.get_shape().second; ++p){
					grad.at(j,p) += X.at(p,i);
					grad.at(k,p) -= X.at(p,i);
				}
			}
	}


	grad *= 1.0 / X.get_shape().second;
	return grad += W * (2 * lambda);
}

double SVM::fit(const Matrix& X, const Matrix& Y){
	for (size_t i = 0; i<steps; ++i){
		cerr << "E: " << setw(5) << i << " | ";
		if (!update(X, Y)) break;
		cerr << endl;
	}
	return loss(W, X,Y);
}

bool SVM::update(const Matrix& X, const Matrix& Y){
	auto grad = grad_loss(W, X, Y);
	cerr << fixed << setprecision(10) << "old_loss: " << setw(18) << loss(W,X,Y) << " | ";
	if (sqrt(grad.square_norm()) < 1e-9) return false;
	vector<double> loss_vals;

	transform(step.begin(), step.end(), back_inserter(loss_vals),
		[&](double s){
			return loss(W + grad * (-s), X, Y);
		});

	size_t pos = min_element(loss_vals.begin(), loss_vals.end()) - loss_vals.begin();
	//cerr << "step: " << setw(11) << step.at(pos) << " | new_loss: " << setw(18) << loss_vals[pos] << " | grad: ";
	//for (size_t i = 0; i<5; ++i)
	//	cerr << grad[i] << " ";
	W += grad *= -step.at(pos);

	return true;
}

}
