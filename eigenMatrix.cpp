#include <iostream>
using namespace std;

// lib related to time operation
#include <ctime>

// main part of Eigen lib
#include <Eigen/Core>

// various matrix operation - inverse, eigenvalue, etc
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50


int main(int argc, char **argv){
	// In Eigen, all vectors and matricies are in class Eigen::Matrix
	// We are already using namespace Eigen, so don't need to type Eigen::
	Matrix<float, 2, 3> matrix_23;
	
	// Also there are a lot of internal type. But in basic, they are all on Matrix class
	Vector3d v_3d;
	Matrix3d matrix_33 = Matrix3d::Zero(); // define Matrix3d and initialize with 0
	
	// We can define size-adjustable matrix
	Matrix<double, Dynamic, Dynamic> matrix_dynamic;
	
	// Initialization of matrix
	matrix_23 << 1, 2, 3, 4, 5, 6;
	
	// print
	cout << "matrix 2x3 from 1 to 6 : \n" << matrix_23 << endl;
	
	// How to access to index
	cout << "print matrix 2x3 : " << endl;
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
		cout << endl;
	}
	
	// Multiplication of matrix
	v_3d << 1, 2, 3;
	
	// If I try to multiplicate matrix_23 and v_3d, it fails
	// matrix_23 has float type and v_3d has double type. so we need to convert type
	Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << "[1, 2, 3; 4, 5, 6]*[1, 2, 3] : " << result.transpose() << endl;
	
	// Matrix operations
	matrix_33 = Matrix3d::Random();
	cout << "random matrix : \n" << matrix_33 << endl;
	cout << "transpose : \n" << matrix_33.transpose() << endl;
	cout << "sum : " << matrix_33.sum() << endl;
	cout << "trace : " << matrix_33.trace() << endl;
	cout << "times 10 : \n" << 10*matrix_33 << endl;
	cout << "inverse : \n" << matrix_33.inverse() << endl;
	cout << "determinant : " << matrix_33.determinant() << endl;
	
	// Eigenvalue
	SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
	cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
	
	// Equation solver
	Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
		= MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
	// to make sure matrix to be semi-positive definite
	matrix_NN = matrix_NN * matrix_NN.transpose();
	Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);
	
	// to check time
	clock_t time_stt = clock();
	
	// calculating inverse matrix
	Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time of normal inverse is " 
		<< 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
	cout << "x = " << x.transpose() << endl;
	
	// QR decomposition
	time_stt = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time of QR decomposition is " 
		<< 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
	cout << "x = " << x.transpose() << endl;
	
	// sholesky decomposition
	time_stt = clock();
	x = matrix_NN.ldlt().solve(v_Nd);
	cout << "time of ldlt is " 
		<< 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
	cout << "x = " << x.transpose() << endl;
	
	//end
	return 0;	
	
	
}
