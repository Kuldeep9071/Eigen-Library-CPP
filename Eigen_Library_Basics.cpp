#include <bits/stdc++.h>
#include "LinearAlgebra/Eigen/Dense"

#define endl "\n"

using namespace std; 
using namespace Eigen;

int main(){
    // Define Static Matrices 
    /* One of the best way is write {Matrix<data_type,m,n> matrix_Name}   where mxn is order of matrix for example a 3x3 matrix*/  

    Matrix<float,3,3> matrixA;
    matrixA.setZero();
    cout << "Matrix A: " << endl << matrixA << endl<<endl;

    /* other way to define square matrix as {Matrixnd} where n is the size of the matrix and
        d is the data type of the matrix example of 3x3 matrix  
        
        d=f for float data type
        d=i for int data type
        d=d for double data type
        d=ld for long double data type

        
    */

    Matrix3d matrixB;
    matrixB.setRandom();
    cout << "Matrix B: " << endl << matrixB << endl<<endl;


    // Define Dynamic Matrices

    /* One of the best way is write {Matrix<data_type,Dynamic,Dynamic> matrix_Name}   where
        Dynamic is the order of matrix defined explicitly for example a 3x3 matrix */

    Matrix<double,Dynamic,Dynamic> matrixC; // Just declaration the matrix & does not allocate memory
    MatrixXd matrixD; // Other way to define the matrix this is also just declaration the matrix & does not allocate memory
    MatrixXd matrixE(2,2); // Using constructor this is declaration and initialization of the matrix with 2x2 order

    // X is the order of the square matrix and d is double data type!

    // Initialize the matrixE with some explicit values
    matrixE(0,0)=1.9;
    matrixE(0,1)=2.9;  // Accessing the elements of the matrix
    matrixE(1,0)=3.9; // M(a-1,b-1) is the element at a-th row and b-th column in 1-based indexing
    matrixE(1,1)=4.9;

    cout << "Matrix E: " << endl << matrixE << endl<<endl; 

    // Filling matrix with entries with comma and space separated values

    MatrixXd matrixF(4,4);
    matrixF <<  1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16;

    cout << "Matrix F: " << endl << matrixF << endl<<endl;

    int rows = 3;
    int cols = 3;

    // Matrix of Zeros
    
    // Matrix of zeros Methdod 1
    MatrixXd matrixG;
    matrixG = MatrixXd::Zero(rows,cols); // Intialisation+Declaration+Memory Allocation

    // Matrix of zeros Method 2
    MatrixXd matrixH(rows,cols); // Declaration of matrix+Memory Allocation
    matrixH.setZero(); 

    // Matrix of zeros Method 3
    MatrixXd matrixI; // Just declaration of matrix
    matrixI.setZero(rows,cols);

    cout << "Matrix G: " << endl << matrixG << endl<<endl;
    cout << "Matrix H: " << endl << matrixH << endl<<endl;
    cout << "Matrix I: " << endl << matrixI << endl<<endl;

    // Matrix of Ones

    // Matrix of Ones Method 1
    MatrixXd matrixJ = MatrixXd::Ones(rows,cols); // Intialisation+Declaration+Memory Allocation

    // Matrix of Ones Method 2
    MatrixXd matrixK(rows,cols); // Declaration of matrix+Memory Allocation
    matrixK.setOnes();

    // Matrix of Ones Method 3
    MatrixXd matrixL; // Just declaration of matrix
    matrixL.setOnes(rows,cols);

    cout << "Matrix J: " << endl << matrixJ << endl<<endl;
    cout << "Matrix K: " << endl << matrixK << endl<<endl;
    cout << "Matrix L: " << endl << matrixL << endl<<endl;

    // Matrix of Constants

    // Matrix of Constants Method 1
    MatrixXd matrixM = MatrixXd::Constant(rows,cols,5); // Intialisation+Declaration+Memory Allocation

    // Matrix of Constants Method 2
    MatrixXd matrixN(rows,cols); // Declaration of matrix+Memory Allocation
    matrixN.setConstant(5);

    // Matrix of Constants Method 3
    MatrixXd matrixO; // Just declaration of matrix
    matrixO.setConstant(rows,cols,5);

    cout << "Matrix M: " << endl << matrixM << endl<<endl;
    cout << "Matrix N: " << endl << matrixN << endl<<endl;
    cout << "Matrix O: " << endl << matrixO << endl<<endl;

    // Identity Matrix

    // Identity Matrix Method 1
    MatrixXd matrixP = MatrixXd::Identity(rows,cols); // Intialisation+Declaration+Memory Allocation

    // Identity Matrix Method 2
    MatrixXd matrixQ(rows,cols); // Declaration of matrix+Memory Allocation
    matrixQ.setIdentity();

    // Identity Matrix Method 3
    MatrixXd matrixR; // Just declaration of matrix
    matrixR.setIdentity(rows,cols);

    cout << "Matrix P: " << endl << matrixP << endl<<endl;
    cout << "Matrix Q: " << endl << matrixQ << endl<<endl;
    cout << "Matrix R: " << endl << matrixR << endl<<endl;


    // Block Matrix

    MatrixXd Parent_Matrix(7,7);
    Parent_Matrix <<  1, 2, 3, 4, 5, 6, 7,
                      8, 9, 10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26, 27, 28,
                      29, 30, 31, 32, 33, 34, 35,
                      36, 37, 38, 39, 40, 41, 42,
                      43, 44, 45, 46, 47, 48, 49;
    
    cout << "Parent Matrix: " << endl << Parent_Matrix << endl<<endl;

    // Block Matrix parent_matrix.block(start_row,start_col,num_rows,num_cols) for example block of 3x3 matrix starting from 1,1 in Parent_Matrix

    MatrixXd Block_Matrix = Parent_Matrix.block(1,1,3,3); // Block of 3x3 matrix starting from 1,1 in Parent_Matrix

    cout << "Block Matrix: " << endl << Block_Matrix << endl<<endl;

    // Vector are just a special case of matrices with only one column or row

    // Vector as diagonal matrix

    Matrix <double,3,1> vectorA;
    vectorA << 1, 2, 3;
    MatrixXd Diagonal_Matrix = vectorA.asDiagonal();

    cout << "Vector A: " << endl << vectorA << endl<<endl;
    cout << "Diagonal Matrix: " << endl << Diagonal_Matrix << endl<<endl;

    // other way to define Vector is to use Vectornd where nx1 is the size of the vector and d is the data type of the vector example of 3x1 vector

    VectorXd vectorB(3); 
    vectorB << 4, 5, 6;

    cout << "Vector B: " << endl << vectorB<< endl<<endl;

    // Matrix Operations

    // Addition

    MatrixXd matrixS = matrixG + matrixH;
    cout << "Matrix S: " << endl << matrixS << endl<<endl;

    // Subtraction

    MatrixXd matrixT = matrixG - matrixH;
    cout << "Matrix T: " << endl << matrixT << endl<<endl;

    // Multiplication

    MatrixXd matrixU = matrixG * matrixH;
    cout << "Matrix U: " << endl << matrixU << endl<<endl;

    // Multiplication with scalar

    MatrixXd matrixV = matrixJ * 2;
    cout<<"Matrix V: "<<endl<<matrixV<<endl<<endl;

    // Transpose : This is Not Inplace

    MatrixXd matrixW = matrixE.transpose();
    cout << "Matrix W: " << endl << matrixW << endl<<endl;

    // Inplace Transpose

    matrixE.transposeInPlace(); // E got Transposed in itseld without creating new matrix or using new memory
    cout << "Matrix E: " << endl << matrixE << endl<<endl;

    // Inverse

    MatrixXd matrixX = matrixB.inverse();
    cout << "Matrix X: " << endl << matrixX << endl<<endl;

    // Determinant

    double determinant = matrixB.determinant();
    cout << "Determinant of Matrix B: " << determinant << endl<<endl;

    // Trace

    double trace = matrixB.trace();
    cout << "Trace of Matrix B: " << trace << endl<<endl;

    // Norm : This calulates the Frobenius norm of the matrix which is the square root of the sum of the squares of the elements of the matrix

    double norm = matrixB.norm();
    cout << "Norm of Matrix B: " << norm << endl<<endl;

}
