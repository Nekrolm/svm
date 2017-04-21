#ifndef INC_MATRIX_H_
#define INC_MATRIX_H_

#include <vector>
#include <cstring>
#include <iostream>
namespace Numeric{

class Matrix {
public:
	Matrix(size_t n = 0, size_t m = 0, double val = 0);
	Matrix(const std::pair<size_t, size_t>& shape);
	Matrix(const Matrix& m) = default;
	Matrix(Matrix&& m);

	~Matrix();

static Matrix get_matrix_from(std::istream& in, std::pair<size_t, size_t> shape);

inline std::pair<size_t, size_t> get_shape() const{
	return shape;
}

inline Matrix& reshape(std::pair<size_t, size_t> shape){
	this->shape = shape;
	return *this;
}

Matrix& append(const Matrix& m);

Matrix& load_matrix_from(std::istream& in, std::pair<size_t, size_t> shape);

double& at(size_t i, size_t j);
double at(size_t i, size_t j) const;
double& operator[](size_t ofs);
double operator[](size_t ofs) const;

Matrix& operator += (const Matrix& other);
Matrix operator + (const Matrix& other) const;
Matrix& operator *= (double k);
Matrix operator * (double k) const;
Matrix& operator *= (const Matrix& other);
Matrix operator * (const Matrix& other) const;

Matrix& operator = (const Matrix &) = default;

double square_norm() const;

Matrix& transpose();

friend std::ostream& operator << (std::ostream& out, const Matrix& m);

inline std::vector<double>::iterator begin(){
	return data.begin();
}

inline std::vector<double>::iterator end(){
	return data.end();
}

private:
	std::pair<size_t, size_t> shape;
	std::vector<double> data;
};

}
#endif /* INC_MATRIX_H_ */
