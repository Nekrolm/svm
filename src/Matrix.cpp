#include "../inc/Matrix.h"
#include "stdexcept"
#include <string>
#include <algorithm>
#include <tuple>
#include <thread>
#include <numeric>
#include <iterator>

using namespace std;
namespace Numeric{

Matrix::Matrix(size_t n, size_t m, double val) :
		shape(n,m), data(n*m, val)
{}

Matrix::Matrix(Matrix&& m){
	data = move(m.data);
	shape = m.shape;
}

Matrix::Matrix(const std::pair<size_t, size_t>& shape) :
		Matrix(shape.first, shape.second)
{
}

Matrix::~Matrix() {
	// TODO Auto-generated destructor stub
}

double& Matrix::at(size_t i, size_t j){
	if (i >= shape.first || j >= shape.second)
		throw range_error("(" + to_string(i) + "," + to_string(j) +
							") cell is not in matrix " +
						to_string(shape.first) + "x" + to_string(shape.first));
	return data[i * shape.second + j];
}

double Matrix::at(size_t i, size_t j) const{
	return const_cast<Matrix*>(this)->at(i,j);
}

double& Matrix::operator [](size_t ofs){
	return data.at(ofs);
}
double Matrix::operator [](size_t ofs) const{
	return data.at(ofs);
}

double Matrix::square_norm() const{
	return inner_product(data.begin(), data.end(), data.begin(), 0.0);
}

Matrix& Matrix::operator += (const Matrix& other){
	transform(data.begin(), data.end(), other.data.begin(),
							data.begin(), plus<double>());
	return *this;
}

Matrix Matrix::operator + (const Matrix& other) const{
	Matrix ret = *this;
	return ret += other;
}


Matrix& Matrix::operator *= (double k){
	transform(data.begin(), data.end(),
			data.begin(), bind2nd(multiplies<double>(),k));
	return *this;
}

Matrix Matrix::operator * (double k) const{
	Matrix ret = *this;
	return ret *= k;
}

Matrix& Matrix::operator *= (const Matrix& other){
	return *this = *this * other;
}

void mul(const Matrix& A, const Matrix& B, Matrix& C,
		size_t start_line, size_t end_line){
	for (size_t i = start_line; i < end_line; ++i){
		for (size_t j = 0; j < B.get_shape().second; ++j){
			double& tmp = C.at(i,j);
			for (size_t k = 0; k < B.get_shape().first; ++k)
				tmp += A.at(i,k) * B.at(k,j);
		}
	}
}

Matrix Matrix::operator * (const Matrix& other) const{
	if (shape.second != other.shape.first)
			throw range_error("Matrix is not aligned");

	Matrix ret(shape.first, other.shape.second);


#ifdef PARALLEL
	int p = thread::hardware_concurrency();
	int sz = shape.first / p;

	vector<thread> thrs(0);
	for (int i = 0; i<p; ++i)
		thrs.emplace_back(mul, ref(*this), ref(other), ref(ret), i*sz, (i+1)*sz);
	mul(*this, other, ret, p*sz, p*sz + shape.first % p);
	for (auto&& t : thrs)
		t.join();
#else
	for (size_t i = 0; i<shape.first; ++i)
		for (size_t j = 0; j<other.shape.second; ++j)
			for (size_t k = 0; k<shape.second; ++k)
				ret.at(i,j) += at(i,k) * other.at(k,j);
#endif
	return ret;
}

Matrix& Matrix::load_matrix_from(istream& in, pair<size_t, size_t> shape){
	this->shape = shape;
	size_t n = shape.first * shape.second;
	data.resize(n);
	for (double& val : data)
		in >> val;
	return *this;
}

ostream& operator << (ostream& out, const Matrix& m){
	auto it = m.data.begin();
	for (size_t i = 0; i < m.shape.first; ++i){
		copy(it, it+m.shape.second, ostream_iterator<double>(out, " "));
		it += m.shape.second;
		out << endl;
	}
	return out;
}

Matrix& Matrix::transpose(){
	size_t n,m;
	tie(n,m) = shape;
	vector<double> tmp(n*m);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < m; ++j)
			tmp[j*n + i] = this->at(i,j);

	data = move(tmp);
	swap(shape.first, shape.second);
	return *this;
}

Matrix& Matrix::append(const Matrix& m){
	shape = {1, shape.first * shape.second + m.shape.first * m.shape.second};
	copy(m.data.begin(), m.data.end(), back_inserter(data));
	return *this;
}

Matrix Matrix::get_matrix_from(istream& in, pair<size_t, size_t> shape){
	return (Matrix().load_matrix_from(in, shape));
}

}
