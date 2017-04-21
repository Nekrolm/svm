#ifndef INC_SVM_H_
#define INC_SVM_H_

#include "Matrix.h"

namespace Classificator{
using Numeric::Matrix;

class SVM {
public:
	SVM(Matrix&& W0, double lambda, size_t steps);
	~SVM();
	double fit(const Matrix& X, const Matrix& Y);
	Matrix predict(const Matrix& x);

inline Matrix getW() const{
	return W;
}

private:
	bool update(const Matrix& X, const Matrix& Y);
	double loss(const Matrix& W, const Matrix& X, const Matrix& Y) const;
	Matrix grad_loss(const Matrix& W, const Matrix& X, const Matrix& Y) const;

	Matrix W;
	double lambda;
	size_t steps;
	std::vector<double> step;
};

}
#endif /* INC_SVM_H_ */
